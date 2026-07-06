import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.scripts.play_hierarchical_nav import (
    CREATE_VIDEO,
    VIDEO_FRAME_STRIDE,
    _compose_low_level_actions,
    _create_video_recorder,
    _load_jit_policy,
    _write_video_frame,
)
from legged_gym.utils.task_registry import get_args, task_registry


PROBE_FORWARD_COMMAND = 0.25
PROBE_OBSTACLE_X = 3.5
PROBE_OBSTACLE_RADIUS = 1.0


def _configure_collision_probe(env_cfg):
    nav_cfg = env_cfg.navigation

    env_cfg.env.num_envs = 1
    env_cfg.env.is_train = True
    env_cfg.env.enable_camera_sensors = CREATE_VIDEO
    env_cfg.env.episode_length_s = 30

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
    env_cfg.commands.trap_time = 1000

    nav_cfg.use_terrain_mesh_obstacles = True
    nav_cfg.path_type = "line"
    nav_cfg.terrain_obstacle_height = 1.2
    nav_cfg.terrain_obstacle_layout = [
        [PROBE_OBSTACLE_X, nav_cfg.path_center_y, PROBE_OBSTACLE_RADIUS],
    ]

    nav_cfg.terminate_on_body_contact = False
    nav_cfg.terminate_on_trap = False
    nav_cfg.terminate_on_collision = False
    nav_cfg.terminate_on_goal = False
    nav_cfg.terminate_on_out_of_bounds = False


def _apply_probe_command(env):
    env.nav_command_buffer[:, :] = 0.0
    env.nav_command_buffer[:, 0] = PROBE_FORWARD_COMMAND
    env.commands[:, :3] = env.nav_command_buffer


def play(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    _configure_collision_probe(env_cfg)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    _apply_probe_command(env)
    env.compute_observations()

    obstacle_pos = env.obstacle_positions[0, 0].detach().cpu().numpy()
    obstacle_radius = env.obstacle_radii[0, 0].item()
    print(
        "collision probe: fixed forward command, navigation policy disabled | "
        f"obstacle=(x={obstacle_pos[0]:.2f}, y={obstacle_pos[1]:.2f}, r={obstacle_radius:.2f})"
    )

    video_writer = None
    video_path = None
    camera_handle = None
    if CREATE_VIDEO:
        camera_handle, video_writer, video_path = _create_video_recorder(
            env, "roll_robot_r_hierarchical_nav_collision_probe"
        )

    locomotion_policy = _load_jit_policy(env_cfg.navigation.locomotion_policy_path, env.device, "Locomotion")
    recovery_policy = _load_jit_policy(env_cfg.navigation.recovery_policy_path, env.device, "Recovery")

    max_steps = int(env.max_episode_length.item()) if hasattr(env.max_episode_length, "item") else int(env.max_episode_length)

    try:
        for step in range(max_steps):
            with torch.inference_mode():
                _apply_probe_command(env)
                low_level_actions = _compose_low_level_actions(env, locomotion_policy, recovery_policy)
                _, _, _, _, _, _ = env.step(low_level_actions)

            if (
                video_writer is not None
                and camera_handle is not None
                and step % VIDEO_FRAME_STRIDE == 0
            ):
                frame_bgr = _write_video_frame(env, camera_handle)
                video_writer.write(frame_bgr)

            if step % 25 == 0:
                status = env.get_navigation_status(0)
                speed = torch.norm(env.base_lin_vel[0, :2]).item()
                print(
                    f"step={step:04d} "
                    f"x={status['local_x']:.2f} "
                    f"y={status['local_y']:.2f} "
                    f"clearance={status['min_clearance']:.2f} "
                    f"logic_collision={status['collision']} "
                    f"speed={speed:.2f} "
                    f"cmd_vx={env.commands[0, 0].item():.2f}"
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
