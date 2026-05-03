import torch

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.scripts.play_hierarchical_nav import (
    CREATE_VIDEO,
    VIDEO_FRAME_STRIDE,
    _compose_low_level_actions,
    _create_video_recorder,
    _load_jit_policy,
    _write_video_frame,
)
from legged_gym.utils.nav_policy_loader import load_navigation_policy
from legged_gym.utils.task_registry import get_args, task_registry


def _configure_terrain_obstacle_demo(env_cfg):
    nav_cfg = env_cfg.navigation

    env_cfg.env.num_envs = 1
    env_cfg.env.is_train = True
    env_cfg.env.enable_camera_sensors = CREATE_VIDEO
    env_cfg.env.episode_length_s = 180

    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.terrain_length = max(
        float(env_cfg.terrain.terrain_length),
        nav_cfg.field_length + nav_cfg.terrain_nav_start_margin + nav_cfg.terrain_nav_end_margin,
    )
    env_cfg.terrain.terrain_width = max(
        float(env_cfg.terrain.terrain_width),
        nav_cfg.field_width + 2.0 * nav_cfg.terrain_nav_side_margin,
    )

    env_cfg.commands.curriculum = False
    env_cfg.commands.trap_time = 30
    nav_cfg.use_terrain_mesh_obstacles = True
    nav_cfg.terminate_on_body_contact = False
    nav_cfg.terminate_on_trap = False
    nav_cfg.terminate_on_collision = True


def play(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    _configure_terrain_obstacle_demo(env_cfg)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    video_writer = None
    video_path = None
    camera_handle = None
    if CREATE_VIDEO:
        camera_handle, video_writer, video_path = _create_video_recorder(
            env, "roll_robot_r_hierarchical_nav_terrain"
        )

    locomotion_policy = _load_jit_policy(env_cfg.navigation.locomotion_policy_path, env.device, "Locomotion")
    recovery_policy = _load_jit_policy(env_cfg.navigation.recovery_policy_path, env.device, "Recovery")
    nav_policy, _ = load_navigation_policy(env_cfg.navigation.nav_policy_path, env.device)

    print("terrain obstacle mode: static trimesh obstacles are synced into navigation observations")

    nav_obs = env.get_nav_observations()
    max_steps = int(env.max_episode_length.item()) if hasattr(env.max_episode_length, "item") else int(env.max_episode_length)
    last_nav_action = torch.zeros(2, device=env.device)

    try:
        for step in range(max_steps):
            with torch.inference_mode():
                nav_actions = nav_policy.act(nav_obs)
                last_nav_action = nav_actions[0].detach().clone()
                env.apply_navigation_actions(nav_actions)
                low_level_actions = _compose_low_level_actions(env, locomotion_policy, recovery_policy)
                nav_obs, _, nav_rewards, nav_dones, _, _ = env.step(low_level_actions)

            if (
                video_writer is not None
                and camera_handle is not None
                and step % VIDEO_FRAME_STRIDE == 0
            ):
                frame_bgr = _write_video_frame(env, camera_handle)
                video_writer.write(frame_bgr)

            if step % 50 == 0:
                status = env.get_navigation_status(0)
                command = env.nav_command_buffer[0]
                nav_action = last_nav_action
                print(
                    f"step={step:04d} "
                    f"x={status['local_x']:.2f} "
                    f"y={status['local_y']:.2f} "
                    f"goal_dist={status['goal_distance']:.2f} "
                    f"track_err={status['track_error']:.2f} "
                    f"clearance={status['min_clearance']:.2f} "
                    f"front={status['front_clearance']:.2f} "
                    f"blend={status['path_blend']:.2f} "
                    f"speed_scale={status['speed_scale']:.2f} "
                    f"nav=(rudder={nav_action[0].item():.2f}, thrust={nav_action[1].item():.2f}) "
                    f"cmd=(vx={command[0].item():.2f}, yaw={command[2].item():.2f})"
                )

            if bool(nav_dones[0].item()):
                status = env.get_navigation_status(0)
                reasons = env.get_termination_status(0)
                command = env.nav_command_buffer[0]
                nav_action = last_nav_action
                print(
                    "episode reset | "
                    f"goal={reasons['goal_reached']} "
                    f"collision={reasons['collision']} "
                    f"out_of_bounds={reasons['out_of_bounds']} "
                    f"body_contact={reasons['body_contact']} "
                    f"trap={reasons['trap']} "
                    f"timeout={reasons['timeout']} "
                    f"front={status['front_clearance']:.2f} "
                    f"blend={status['path_blend']:.2f} "
                    f"speed_scale={status['speed_scale']:.2f} "
                    f"nav=(rudder={nav_action[0].item():.2f}, thrust={nav_action[1].item():.2f}) "
                    f"cmd=(vx={command[0].item():.2f}, yaw={command[2].item():.2f}) "
                    f"reward={nav_rewards[0].item():.2f}"
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
