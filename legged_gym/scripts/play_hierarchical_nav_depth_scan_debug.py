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
CAMERA_FORWARD_OFFSET = 0.35
CAMERA_UP_OFFSET = 0.35
CAMERA_PITCH_TARGET_DROP = 0.08


def _create_depth_camera(env, scanner_config):
    camera_props = gymapi.CameraProperties()
    camera_props.width = scanner_config.camera_width
    camera_props.height = scanner_config.camera_height
    camera_props.horizontal_fov = scanner_config.horizontal_fov_deg
    camera_handle = env.gym.create_camera_sensor(env.envs[SCAN_TRACK_ENV], camera_props)
    if camera_handle == -1:
        raise RuntimeError("Failed to create depth camera sensor.")
    return camera_handle


def _camera_pose_from_robot(env):
    root_pos = env.root_states[SCAN_TRACK_ENV, :3].detach().cpu().numpy()
    heading = float(env.nav_heading[SCAN_TRACK_ENV].item())
    forward = np.array([np.cos(heading), np.sin(heading), 0.0], dtype=np.float32)

    camera_pos = root_pos + CAMERA_FORWARD_OFFSET * forward
    camera_pos[2] += CAMERA_UP_OFFSET
    target_pos = camera_pos + forward
    target_pos[2] -= CAMERA_PITCH_TARGET_DROP
    return camera_pos.astype(np.float32), target_pos.astype(np.float32)


def _read_depth_image(env, camera_handle, scanner_config):
    camera_pos, target_pos = _camera_pose_from_robot(env)
    env.gym.set_camera_location(
        camera_handle,
        env.envs[SCAN_TRACK_ENV],
        gymapi.Vec3(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])),
        gymapi.Vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
    )

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
            camera_scan = scanner.compute("camera", depth_image=depth_image)
            oracle_scan = _compute_oracle_scan(env, scanner)

            camera_summary = scanner.summarize(camera_scan)
            oracle_summary = scanner.summarize(oracle_scan)
            status = env.get_navigation_status(SCAN_TRACK_ENV)
            print(
                f"step={step:04d} "
                f"x={status['local_x']:.2f} "
                f"y={status['local_y']:.2f} "
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
