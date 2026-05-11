import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.scripts.play_hierarchical_nav import _compose_low_level_actions, _load_jit_policy
from legged_gym.scripts.play_hierarchical_nav_terrain import _configure_terrain_obstacle_demo
from legged_gym.utils.nav_depth_scan import DepthScanConfig, NavDepthScanner
from legged_gym.utils.nav_policy_loader import load_navigation_policy
from legged_gym.utils.task_registry import get_args, task_registry


SCAN_PRINT_STRIDE = 50
SCAN_TRACK_ENV = 0
CAMERA_MOUNT_BODY = "base_link"
CAMERA_LOCAL_POSITION = np.array([0.23, 0.0, 0.16], dtype=np.float32)
CAMERA_MOUNT_PITCH_DEG = 0.0


def _create_depth_camera(env, scanner_config):
    camera_props = gymapi.CameraProperties()
    camera_props.width = scanner_config.camera_width
    camera_props.height = scanner_config.camera_height
    camera_props.horizontal_fov = scanner_config.horizontal_fov_deg
    camera_handle = env.gym.create_camera_sensor(env.envs[SCAN_TRACK_ENV], camera_props)
    if camera_handle == -1:
        raise RuntimeError("Failed to create depth camera sensor.")

    body_handle = env.gym.find_actor_rigid_body_handle(
        env.envs[SCAN_TRACK_ENV],
        env.actor_handles[SCAN_TRACK_ENV],
        CAMERA_MOUNT_BODY,
    )
    if body_handle == -1:
        raise RuntimeError(f"Camera mount body not found: {CAMERA_MOUNT_BODY}")

    local_transform = gymapi.Transform()
    local_transform.p = gymapi.Vec3(
        float(CAMERA_LOCAL_POSITION[0]),
        float(CAMERA_LOCAL_POSITION[1]),
        float(CAMERA_LOCAL_POSITION[2]),
    )
    local_transform.r = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(0.0, 1.0, 0.0),
        float(np.deg2rad(CAMERA_MOUNT_PITCH_DEG)),
    )
    env.gym.attach_camera_to_body(
        camera_handle,
        env.envs[SCAN_TRACK_ENV],
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
    root_pos = env.root_states[SCAN_TRACK_ENV, :3].detach().cpu().numpy()
    root_quat = env.root_states[SCAN_TRACK_ENV, 3:7].detach().cpu().numpy()

    pitch = float(np.deg2rad(CAMERA_MOUNT_PITCH_DEG))
    local_forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)], dtype=np.float32)
    local_right = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    local_up = np.array([np.sin(pitch), 0.0, np.cos(pitch)], dtype=np.float32)

    camera_pos = root_pos + _quat_rotate(root_quat, CAMERA_LOCAL_POSITION)
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
        env.envs[SCAN_TRACK_ENV],
        camera_handle,
        gymapi.IMAGE_DEPTH,
    )
    if depth_image is None or len(depth_image) == 0:
        raise RuntimeError("Depth camera returned an empty image.")
    return np.reshape(depth_image, (scanner_config.camera_height, scanner_config.camera_width))


def _format_summary(prefix, summary):
    band_front = ",".join(f"{value:.2f}" for value in summary["band_front_min"])
    return (
        f"{prefix}: "
        f"front={summary['front_min']:.2f} "
        f"left={summary['left_min']:.2f} "
        f"right={summary['right_min']:.2f} "
        f"min={summary['overall_min']:.2f}@{summary['overall_min_angle_deg']:.1f}deg "
        f"band_front=[{band_front}]"
    )


def _compute_oracle_scan(env, scanner):
    env_id = SCAN_TRACK_ENV
    return scanner.compute(
        "oracle",
        sensor_xy=env.nav_local_pos[env_id].detach().cpu().numpy(),
        heading=float(env.nav_heading[env_id].item()),
        obstacle_positions=env.obstacle_positions[env_id].detach().cpu().numpy(),
        obstacle_radii=env.obstacle_radii[env_id].detach().cpu().numpy(),
    )


def play(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    _configure_terrain_obstacle_demo(env_cfg)
    env_cfg.env.enable_camera_sensors = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    scanner_config = DepthScanConfig()
    scanner = NavDepthScanner(scanner_config)
    depth_camera = _create_depth_camera(env, scanner_config)

    locomotion_policy = _load_jit_policy(env_cfg.navigation.locomotion_policy_path, env.device, "Locomotion")
    recovery_policy = _load_jit_policy(env_cfg.navigation.recovery_policy_path, env.device, "Recovery")
    nav_policy, _ = load_navigation_policy(env_cfg.navigation.nav_policy_path, env.device)

    print(
        "depth-scan debug: oracle and camera scans are computed side by side; "
        "they are not fed into the navigation policy yet"
    )

    nav_obs = env.get_nav_observations()
    max_steps = int(env.max_episode_length.item()) if hasattr(env.max_episode_length, "item") else int(env.max_episode_length)

    for step in range(max_steps):
        with torch.inference_mode():
            nav_actions = nav_policy.act(nav_obs)
            env.apply_navigation_actions(nav_actions)
            low_level_actions = _compose_low_level_actions(env, locomotion_policy, recovery_policy)
            nav_obs, _, _, nav_dones, _, _ = env.step(low_level_actions)

        if step % SCAN_PRINT_STRIDE == 0:
            depth_image = _read_depth_image(env, depth_camera, scanner_config)
            camera_pos, camera_forward, camera_right, camera_up = _attached_camera_frame(env)
            ground_height = float(env.env_origins[SCAN_TRACK_ENV, 2].item())
            camera_scan = scanner.compute(
                "camera",
                depth_image=depth_image,
                camera_position=camera_pos,
                camera_forward=camera_forward,
                camera_right=camera_right,
                camera_up=camera_up,
                ground_height=ground_height,
            )
            oracle_scan = _compute_oracle_scan(env, scanner)

            camera_summary = scanner.summarize(camera_scan)
            oracle_summary = scanner.summarize(oracle_scan)
            status = env.get_navigation_status(SCAN_TRACK_ENV)
            print(
                f"step={step:04d} "
                f"x={status['local_x']:.2f} "
                f"y={status['local_y']:.2f} "
                f"cam_z={camera_pos[2]:.2f} "
                f"{_format_summary('camera', camera_summary)} | "
                f"{_format_summary('oracle', oracle_summary)}"
            )

        if bool(nav_dones[SCAN_TRACK_ENV].item()):
            reasons = env.get_termination_status(SCAN_TRACK_ENV)
            print(
                "episode reset | "
                f"goal={reasons['goal_reached']} "
                f"collision={reasons['collision']} "
                f"out_of_bounds={reasons['out_of_bounds']} "
                f"timeout={reasons['timeout']}"
            )


if __name__ == "__main__":
    args = get_args()
    args.task = "roll_robot_r_hierarchical_nav"
    args.headless = True
    play(args)
