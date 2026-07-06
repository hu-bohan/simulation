import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.scripts.play_hierarchical_nav import (
    CREATE_VIDEO,
    VIDEO_FRAME_STRIDE,
    _compose_low_level_actions,
    _create_video_recorder,
    _load_jit_policy,
    _write_video_frame,
)
from legged_gym.scripts.hierarchical_nav_demo_utils import (
    create_depth_nav_translator,
    create_depth_scan_config,
    depth_camera_local_frame,
    depth_camera_mount_position,
    depth_camera_mount_quat,
)
from legged_gym.scripts.play_hierarchical_nav import _configure_terrain_obstacle_demo
from legged_gym.utils.nav_depth_scan import NavDepthScanner
from legged_gym.utils.nav_policy_loader import load_navigation_policy
from legged_gym.utils.task_registry import get_args, task_registry


DEPTH_PRINT_STRIDE = 50
DEPTH_TRACK_ENV = 0


def _create_depth_camera(env, scanner_config):
    depth_cfg = env.cfg.navigation.depth_camera
    camera_props = gymapi.CameraProperties()
    camera_props.width = scanner_config.camera_width
    camera_props.height = scanner_config.camera_height
    camera_props.horizontal_fov = scanner_config.horizontal_fov_deg
    camera_handle = env.gym.create_camera_sensor(env.envs[DEPTH_TRACK_ENV], camera_props)
    if camera_handle == -1:
        raise RuntimeError("Failed to create depth camera sensor.")

    body_handle = env.gym.find_actor_rigid_body_handle(
        env.envs[DEPTH_TRACK_ENV],
        env.actor_handles[DEPTH_TRACK_ENV],
        depth_cfg.mount_body,
    )
    if body_handle == -1:
        raise RuntimeError(f"Camera mount body not found: {depth_cfg.mount_body}")

    local_position = depth_camera_mount_position(env.cfg.navigation)
    local_transform = gymapi.Transform()
    local_transform.p = gymapi.Vec3(
        float(local_position[0]),
        float(local_position[1]),
        float(local_position[2]),
    )
    local_transform.r = depth_camera_mount_quat(gymapi, env.cfg.navigation)
    env.gym.attach_camera_to_body(
        camera_handle,
        env.envs[DEPTH_TRACK_ENV],
        body_handle,
        local_transform,
        gymapi.FOLLOW_TRANSFORM,
    )
    return camera_handle


def _quat_rotate(quat_xyzw, vector):
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    xyz = quat[:3]
    w = quat[3]
    uv = np.cross(xyz, vector)
    uuv = np.cross(xyz, uv)
    return vector + 2.0 * (w * uv + uuv)


def _attached_camera_frame(env):
    root_pos = env.root_states[DEPTH_TRACK_ENV, :3].detach().cpu().numpy()
    root_quat = env.root_states[DEPTH_TRACK_ENV, 3:7].detach().cpu().numpy()

    local_position = depth_camera_mount_position(env.cfg.navigation)
    local_forward, local_right, local_up = depth_camera_local_frame(env.cfg.navigation)

    camera_pos = root_pos + _quat_rotate(root_quat, local_position)
    camera_forward = _quat_rotate(root_quat, local_forward)
    camera_right = _quat_rotate(root_quat, local_right)
    camera_up = _quat_rotate(root_quat, local_up)
    return camera_pos.astype(np.float32), camera_forward, camera_right, camera_up


def _read_depth_image(env, camera_handle, scanner_config):
    if env.device != "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)

    depth_image = env.gym.get_camera_image(
        env.sim,
        env.envs[DEPTH_TRACK_ENV],
        camera_handle,
        gymapi.IMAGE_DEPTH,
    )
    if depth_image is None or len(depth_image) == 0:
        raise RuntimeError("Depth camera returned an empty image.")
    return np.reshape(depth_image, (scanner_config.camera_height, scanner_config.camera_width))


def _create_translator(env, scanner):
    return create_depth_nav_translator(env.cfg.navigation, scanner)


def _make_depth_nav_observation(env, depth_camera, scanner_config, translator):
    depth_image = _read_depth_image(env, depth_camera, scanner_config)
    camera_pos, camera_forward, camera_right, camera_up = _attached_camera_frame(env)
    ground_height = float(env.env_origins[DEPTH_TRACK_ENV, 2].item())
    base_nav_obs = env.get_nav_observations()[DEPTH_TRACK_ENV].detach().cpu().numpy()

    translation = translator.translate(
        base_nav_obs=base_nav_obs,
        depth_image=depth_image,
        camera_position=camera_pos,
        camera_forward=camera_forward,
        camera_right=camera_right,
        camera_up=camera_up,
        ground_height=ground_height,
    )
    nav_obs = torch.as_tensor(translation.nav_obs, dtype=torch.float32, device=env.device).unsqueeze(0)
    return nav_obs, translation


def _apply_navigation_actions_with_depth_clearance(env, nav_actions, front_clearance):
    nav_actions = torch.clamp(nav_actions, -1.0, 1.0).to(env.device)
    if nav_actions.shape[0] != env.num_envs:
        if env.num_envs == 1 and nav_actions.shape[0] >= 1:
            nav_actions = nav_actions[:1]
        else:
            raise ValueError("nav_actions batch size must match env.num_envs.")

    front_clearance = torch.as_tensor(front_clearance, dtype=torch.float32, device=env.device)
    if front_clearance.ndim == 0:
        front_clearance = front_clearance.repeat(env.num_envs)
    front_clearance = front_clearance.reshape(-1)
    if front_clearance.shape[0] != env.num_envs:
        raise ValueError("front_clearance batch size must match env.num_envs.")

    nav_cfg = env.cfg.navigation
    thrust_norm = (nav_actions[:, 1] + 1.0) * 0.5
    target_commands = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float)
    target_commands[:, 0] = nav_cfg.min_forward_command + thrust_norm * (
        nav_cfg.max_forward_command - nav_cfg.min_forward_command
    )
    target_commands[:, 0] = torch.where(
        thrust_norm < 0.05,
        torch.zeros_like(target_commands[:, 0]),
        target_commands[:, 0],
    )

    policy_yaw = nav_actions[:, 0] * nav_cfg.max_yaw_command
    target_yaw, path_blend = env._blend_path_following_yaw(policy_yaw, front_clearance)
    speed_scale = env._compute_forward_speed_scale(target_yaw, front_clearance)

    target_commands[:, 0] *= speed_scale
    cruise_allowed = (front_clearance > nav_cfg.cruise_clearance) & (~env.nav_reached_goal_buf)
    cruise_command = torch.full_like(target_commands[:, 0], nav_cfg.cruise_forward_command)
    target_commands[:, 0] = torch.where(
        cruise_allowed,
        torch.maximum(target_commands[:, 0], cruise_command),
        target_commands[:, 0],
    )
    target_commands[:, 2] = target_yaw

    linear_smoothing = getattr(nav_cfg, "linear_command_smoothing", nav_cfg.command_smoothing)
    yaw_smoothing = getattr(nav_cfg, "yaw_command_smoothing", nav_cfg.command_smoothing)
    env.nav_command_buffer[:, :2] = (
        linear_smoothing * env.nav_command_buffer[:, :2]
        + (1.0 - linear_smoothing) * target_commands[:, :2]
    )
    env.nav_command_buffer[:, 2] = (
        yaw_smoothing * env.nav_command_buffer[:, 2]
        + (1.0 - yaw_smoothing) * target_commands[:, 2]
    )
    env.nav_command_buffer[env.nav_reached_goal_buf] = 0.0
    env.nav_last_actions[:] = nav_actions
    env.nav_front_clearance[:] = front_clearance
    env.nav_path_follow_blend[:] = path_blend
    env.nav_speed_scale[:] = speed_scale
    env.commands[:, :3] = env.nav_command_buffer


def _format_depth_status(translation):
    if translation.obstacle_count == 0:
        return f"depth_front={translation.front_clearance:.2f} depth_obs=0 nearest=none"

    nearest = translation.obstacles[0]
    nearest_angle = float(np.rad2deg(nearest.angle))
    return (
        f"depth_front={translation.front_clearance:.2f} "
        f"depth_obs={translation.obstacle_count} "
        f"nearest={nearest.surface_distance:.2f}@{nearest_angle:.1f}deg "
        f"r={nearest.radius:.2f}"
    )


def play(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    _configure_terrain_obstacle_demo(env_cfg)
    env_cfg.env.enable_camera_sensors = True
    env_cfg.navigation.depth_camera.enabled = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    scanner_config = create_depth_scan_config(env_cfg.navigation)
    scanner = NavDepthScanner(scanner_config)
    translator = _create_translator(env, scanner)
    depth_camera = _create_depth_camera(env, scanner_config)

    video_writer = None
    video_path = None
    video_camera = None
    if CREATE_VIDEO:
        video_camera, video_writer, video_path = _create_video_recorder(
            env, "roll_robot_r_hierarchical_nav_depth_camera"
        )

    locomotion_policy = _load_jit_policy(env_cfg.navigation.locomotion_policy_path, env.device, "Locomotion")
    recovery_policy = _load_jit_policy(env_cfg.navigation.recovery_policy_path, env.device, "Recovery")
    nav_policy, _ = load_navigation_policy(env_cfg.navigation.nav_policy_path, env.device)

    print(
        "depth-camera navigation mode: the navigation actor receives camera-derived "
        "obstacle slots; oracle obstacle slots are not used for policy input"
    )

    max_steps = int(env.max_episode_length.item()) if hasattr(env.max_episode_length, "item") else int(env.max_episode_length)
    last_nav_action = torch.zeros(2, device=env.device)

    try:
        for step in range(max_steps):
            nav_obs, translation = _make_depth_nav_observation(
                env,
                depth_camera,
                scanner_config,
                translator,
            )

            with torch.inference_mode():
                nav_actions = nav_policy.act(nav_obs)
                last_nav_action = nav_actions[0].detach().clone()
                _apply_navigation_actions_with_depth_clearance(
                    env,
                    nav_actions,
                    translation.front_clearance,
                )
                low_level_actions = _compose_low_level_actions(env, locomotion_policy, recovery_policy)
                _, _, nav_rewards, nav_dones, _, _ = env.step(low_level_actions)

            if (
                video_writer is not None
                and video_camera is not None
                and step % VIDEO_FRAME_STRIDE == 0
            ):
                frame_bgr = _write_video_frame(env, video_camera)
                video_writer.write(frame_bgr)

            if step % DEPTH_PRINT_STRIDE == 0:
                status = env.get_navigation_status(DEPTH_TRACK_ENV)
                command = env.nav_command_buffer[DEPTH_TRACK_ENV]
                nav_action = last_nav_action
                print(
                    f"step={step:04d} "
                    f"x={status['local_x']:.2f} "
                    f"y={status['local_y']:.2f} "
                    f"goal_dist={status['goal_distance']:.2f} "
                    f"track_err={status['track_error']:.2f} "
                    f"{_format_depth_status(translation)} "
                    f"blend={status['path_blend']:.2f} "
                    f"speed_scale={status['speed_scale']:.2f} "
                    f"nav=(rudder={nav_action[0].item():.2f}, thrust={nav_action[1].item():.2f}) "
                    f"cmd=(vx={command[0].item():.2f}, yaw={command[2].item():.2f})"
                )

            if bool(nav_dones[DEPTH_TRACK_ENV].item()):
                status = env.get_navigation_status(DEPTH_TRACK_ENV)
                reasons = env.get_termination_status(DEPTH_TRACK_ENV)
                command = env.nav_command_buffer[DEPTH_TRACK_ENV]
                nav_action = last_nav_action
                print(
                    "episode reset | "
                    f"goal={reasons['goal_reached']} "
                    f"collision={reasons['collision']} "
                    f"out_of_bounds={reasons['out_of_bounds']} "
                    f"body_contact={reasons['body_contact']} "
                    f"trap={reasons['trap']} "
                    f"timeout={reasons['timeout']} "
                    f"{_format_depth_status(translation)} "
                    f"blend={status['path_blend']:.2f} "
                    f"speed_scale={status['speed_scale']:.2f} "
                    f"nav=(rudder={nav_action[0].item():.2f}, thrust={nav_action[1].item():.2f}) "
                    f"cmd=(vx={command[0].item():.2f}, yaw={command[2].item():.2f}) "
                    f"reward={nav_rewards[DEPTH_TRACK_ENV].item():.2f}"
                )
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"video saved to: {video_path}")


if __name__ == "__main__":
    args = get_args()
    args.task = "roll_robot_r_hierarchical_nav"
    args.headless = True
    play(args)
