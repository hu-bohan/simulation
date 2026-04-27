import os
from datetime import datetime

import cv2
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils.nav_policy_loader import load_navigation_policy
from legged_gym.utils.task_registry import get_args, task_registry


CREATE_VIDEO = True
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 50
VIDEO_FRAME_STRIDE = 1
VIDEO_HORIZONTAL_FOV_DEG = 75.0
VIDEO_TRACK_ENV = 0
VIDEO_CAMERA_OFFSET = np.array([-2.8, -1.6, 1.4], dtype=np.float32)
VIDEO_TARGET_OFFSET = np.array([0.4, 0.0, 0.35], dtype=np.float32)


def _load_jit_policy(policy_path, device, label):
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"{label} policy not found: {policy_path}")

    policy = torch.jit.load(policy_path, map_location=device).to(device)
    policy.eval()
    return policy


def _compose_low_level_actions(env, locomotion_policy, recovery_policy):
    student_obs = env.get_student_obs().detach()
    locomotion_actions = locomotion_policy(student_obs)
    protective_mask, recovery_mask = env.get_low_level_masks()

    actions = locomotion_actions
    if protective_mask.any() or recovery_mask.any():
        actions = actions.clone()

    if protective_mask.any():
        protective_actions = env.get_protective_actions()
        actions[protective_mask] = protective_actions[protective_mask]

    if recovery_mask.any():
        recovery_actions = recovery_policy(student_obs)
        actions[recovery_mask] = recovery_actions[recovery_mask]

    return actions


def _normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return vector
    return vector / norm


def _get_camera_pose(env):
    robot_pos = env.root_states[VIDEO_TRACK_ENV, 0:3].detach().cpu().numpy()
    camera_pos = robot_pos + VIDEO_CAMERA_OFFSET
    target_pos = robot_pos + VIDEO_TARGET_OFFSET
    return camera_pos.astype(np.float32), target_pos.astype(np.float32)


def _project_world_points(world_points, camera_pos, target_pos):
    if len(world_points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool), 1.0, 1.0

    forward = _normalize(target_pos - camera_pos)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = _normalize(right)
    up = _normalize(np.cross(right, forward))

    rel = world_points - camera_pos[None, :]
    x_cam = rel @ right
    y_cam = rel @ up
    z_cam = rel @ forward

    h_fov = np.deg2rad(VIDEO_HORIZONTAL_FOV_DEG)
    v_fov = 2.0 * np.arctan(np.tan(h_fov * 0.5) * (VIDEO_HEIGHT / VIDEO_WIDTH))
    fx = VIDEO_WIDTH / (2.0 * np.tan(h_fov * 0.5))
    fy = VIDEO_HEIGHT / (2.0 * np.tan(v_fov * 0.5))

    visible = z_cam > 0.05
    z_safe = np.clip(z_cam, 1e-4, None)
    u = fx * (x_cam / z_safe) + VIDEO_WIDTH * 0.5
    v = VIDEO_HEIGHT * 0.5 - fy * (y_cam / z_safe)
    projected = np.stack([u, v, z_cam], axis=1)
    return projected, visible, fx, fy


def _draw_navigation_overlay(frame_bgr, env, camera_pos, target_pos):
    overlay = frame_bgr.copy()
    origin = env.env_origins[VIDEO_TRACK_ENV].detach().cpu().numpy()
    path_shift_y = env.cfg.navigation.field_width * 0.5

    path_local = env.path_points_local.detach().cpu().numpy()[::2]
    path_world = np.column_stack(
        [
            origin[0] + path_local[:, 0],
            origin[1] + path_local[:, 1] - path_shift_y,
            np.full(path_local.shape[0], origin[2] + 0.05, dtype=np.float32),
        ]
    ).astype(np.float32)

    projected_path, path_visible, _, _ = _project_world_points(path_world, camera_pos, target_pos)
    for idx in range(len(projected_path) - 1):
        if not (path_visible[idx] and path_visible[idx + 1]):
            continue
        p0 = projected_path[idx]
        p1 = projected_path[idx + 1]
        cv2.line(
            overlay,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            (255, 200, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    goal_local = env.goal_local.detach().cpu().numpy()
    goal_world = np.array(
        [[origin[0] + goal_local[0], origin[1] + goal_local[1] - path_shift_y, origin[2] + 0.15]],
        dtype=np.float32,
    )
    projected_goal, goal_visible, fx, _ = _project_world_points(goal_world, camera_pos, target_pos)
    if goal_visible[0]:
        goal_depth = max(projected_goal[0, 2], 0.2)
        goal_radius = max(8, int(round(fx * env.cfg.navigation.goal_tolerance / goal_depth)))
        goal_center = (int(round(projected_goal[0, 0])), int(round(projected_goal[0, 1])))
        cv2.circle(overlay, goal_center, goal_radius, (0, 220, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            "goal",
            (goal_center[0] + 8, goal_center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )

    obstacle_local = env.obstacle_positions[VIDEO_TRACK_ENV].detach().cpu().numpy()
    obstacle_radii = env.obstacle_radii[VIDEO_TRACK_ENV].detach().cpu().numpy()
    obstacle_world = np.column_stack(
        [
            origin[0] + obstacle_local[:, 0],
            origin[1] + obstacle_local[:, 1] - path_shift_y,
            np.full(obstacle_local.shape[0], origin[2] + 0.12, dtype=np.float32),
        ]
    ).astype(np.float32)
    projected_obstacles, obstacle_visible, fx, _ = _project_world_points(
        obstacle_world, camera_pos, target_pos
    )
    for idx, radius in enumerate(obstacle_radii):
        if radius <= 0.0 or not obstacle_visible[idx]:
            continue
        depth = max(projected_obstacles[idx, 2], 0.2)
        pixel_radius = max(6, int(round(fx * radius / depth)))
        center = (int(round(projected_obstacles[idx, 0])), int(round(projected_obstacles[idx, 1])))
        cv2.circle(overlay, center, pixel_radius, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    cv2.putText(
        overlay,
        "path",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 200, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "goal",
        (24, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "obstacles",
        (24, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _create_video_recorder(env, experiment_name):
    video_env = VIDEO_TRACK_ENV
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = VIDEO_WIDTH
    camera_properties.height = VIDEO_HEIGHT
    camera_properties.horizontal_fov = VIDEO_HORIZONTAL_FOV_DEG
    camera_handle = env.gym.create_camera_sensor(env.envs[video_env], camera_properties)
    if camera_handle == -1:
        raise RuntimeError(
            "Failed to create camera sensor. Offscreen rendering needs a graphics device even in headless mode."
        )

    camera_pos, target_pos = _get_camera_pose(env)

    env.gym.set_camera_location(
        camera_handle,
        env.envs[video_env],
        gymapi.Vec3(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])),
        gymapi.Vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
    )
    if env.device != "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)

    first_frame = env.gym.get_camera_image(env.sim, env.envs[video_env], camera_handle, gymapi.IMAGE_COLOR)
    if first_frame is None or len(first_frame) == 0:
        raise RuntimeError("Camera sensor returned an empty frame during recorder initialization.")
    first_frame = np.reshape(first_frame, (VIDEO_HEIGHT, VIDEO_WIDTH, 4))
    frame_size = (first_frame.shape[1], first_frame.shape[0])

    output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "video", experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{datetime.now().strftime('%m%d_%H%M%S')}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, frame_size)
    return camera_handle, writer, output_path


def _write_video_frame(env, camera_handle):
    camera_pos, target_pos = _get_camera_pose(env)

    env.gym.set_camera_location(
        camera_handle,
        env.envs[VIDEO_TRACK_ENV],
        gymapi.Vec3(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])),
        gymapi.Vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
    )
    if env.device != "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    frame = env.gym.get_camera_image(env.sim, env.envs[VIDEO_TRACK_ENV], camera_handle, gymapi.IMAGE_COLOR)
    if frame is None or len(frame) == 0:
        raise RuntimeError("Camera sensor returned an empty frame while writing video.")
    frame = np.reshape(frame, (VIDEO_HEIGHT, VIDEO_WIDTH, 4))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return _draw_navigation_overlay(frame_bgr, env, camera_pos, target_pos)


def play(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.is_train = True
    env_cfg.env.enable_camera_sensors = CREATE_VIDEO
    env_cfg.env.episode_length_s = 180
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.commands.curriculum = False
    env_cfg.commands.trap_time = 30
    env_cfg.navigation.terminate_on_body_contact = False
    env_cfg.navigation.terminate_on_trap = False
    env_cfg.navigation.terminate_on_collision = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    video_writer = None
    video_path = None
    camera_handle = None
    if CREATE_VIDEO:
        camera_handle, video_writer, video_path = _create_video_recorder(
            env, "roll_robot_r_hierarchical_nav"
        )

    locomotion_policy = _load_jit_policy(env_cfg.navigation.locomotion_policy_path, env.device, "Locomotion")
    recovery_policy = _load_jit_policy(env_cfg.navigation.recovery_policy_path, env.device, "Recovery")
    nav_policy, _ = load_navigation_policy(env_cfg.navigation.nav_policy_path, env.device)

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
