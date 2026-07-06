"""Canonical hierarchical navigation demo entry point.

Supported modes:
- oracle: preserve the original baseline using environment-provided obstacle
  slots and the trained locomotion, recovery, and navigation policies.
- terrain: preserve the static terrain/trimesh obstacle demo using
  environment-provided obstacle slots and the trained policies.
- depth-scan-debug: create the simulated D435 depth camera, compare
  camera-derived scans with oracle obstacle geometry, and keep policy input
  on environment-provided obstacle slots.
- depth-camera: feed camera-derived 24D observations to the trained navigation
  actor while keeping the existing policy interface unchanged.
"""

import os
import sys
from datetime import datetime

import cv2
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.scripts.hierarchical_nav_demo_utils import (
    create_depth_nav_translator,
    create_depth_scan_config,
    depth_camera_local_frame,
    depth_camera_mount_position,
    depth_camera_mount_quat,
)
from legged_gym.utils.nav_depth_debug import (
    compare_depth_policy_inputs,
    compare_depth_scans,
    format_depth_policy_comparison,
    format_depth_scan_comparison,
)
from legged_gym.utils.nav_depth_scan import NavDepthScanner
from legged_gym.utils.nav_policy_loader import load_navigation_policy
from legged_gym.utils.task_registry import get_args, task_registry


CANONICAL_NAV_MODES = ("oracle", "terrain", "depth-scan-debug", "depth-camera")
DEFAULT_NAV_MODE = "oracle"
DEPTH_SCAN_PRINT_STRIDE = 50
DEPTH_CAMERA_PRINT_STRIDE = 50

CREATE_VIDEO = True
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 50
VIDEO_FRAME_STRIDE = 1
VIDEO_HORIZONTAL_FOV_DEG = 75.0
VIDEO_TRACK_ENV = 0
VIDEO_CAMERA_OFFSET = np.array([-2.8, -1.6, 1.4], dtype=np.float32)
VIDEO_TARGET_OFFSET = np.array([0.4, 0.0, 0.35], dtype=np.float32)
VIDEO_GROUND_MARKER_Z = 0.025
VIDEO_GROUND_CIRCLE_SEGMENTS = 96
VIDEO_CAMERA_SCAN_STRIDE = 2


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


def _make_ground_circle_points(center_xy, radius, z_value):
    angles = np.linspace(0.0, 2.0 * np.pi, VIDEO_GROUND_CIRCLE_SEGMENTS + 1, dtype=np.float32)
    return np.column_stack(
        [
            center_xy[0] + radius * np.cos(angles),
            center_xy[1] + radius * np.sin(angles),
            np.full_like(angles, z_value),
        ]
    ).astype(np.float32)


def _draw_projected_polyline(frame_bgr, projected_points, visible, color, thickness):
    for idx in range(len(projected_points) - 1):
        if not (visible[idx] and visible[idx + 1]):
            continue
        p0 = projected_points[idx]
        p1 = projected_points[idx + 1]
        cv2.line(
            frame_bgr,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def _draw_projected_points(frame_bgr, projected_points, visible, color, radius):
    for point, is_visible in zip(projected_points, visible):
        if not is_visible:
            continue
        cv2.circle(
            frame_bgr,
            (int(round(point[0])), int(round(point[1]))),
            radius,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )


def _camera_scan_world_points(env, scanner, camera_scan):
    if camera_scan is None or scanner is None:
        return np.zeros((0, 3), dtype=np.float32)

    scan = np.asarray(camera_scan, dtype=np.float32)
    if scan.ndim != 2 or scan.shape[1] != scanner.config.num_rays:
        return np.zeros((0, 3), dtype=np.float32)

    env_id = VIDEO_TRACK_ENV
    origin = env.env_origins[env_id].detach().cpu().numpy()
    shift_y = env.cfg.navigation.field_width * 0.5
    ground_z = origin[2] + VIDEO_GROUND_MARKER_Z
    local_pos = env.nav_local_pos[env_id].detach().cpu().numpy()
    heading = float(env.nav_heading[env_id].item())

    ray_depth = np.min(np.where(np.isfinite(scan), scan, scanner.config.max_depth), axis=0)
    valid = ray_depth < scanner.config.max_depth * 0.98
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    points = []
    for ray_idx in np.where(valid)[0][::VIDEO_CAMERA_SCAN_STRIDE]:
        distance = float(ray_depth[ray_idx])
        angle = heading + float(scanner.config.ray_angles[ray_idx])
        local_x = local_pos[0] + distance * np.cos(angle)
        local_y = local_pos[1] + distance * np.sin(angle)
        points.append([origin[0] + local_x, origin[1] + local_y - shift_y, ground_z])

    return np.asarray(points, dtype=np.float32)


def _draw_navigation_overlay(frame_bgr, env, camera_pos, target_pos, camera_scan=None, scanner=None):
    overlay = frame_bgr.copy()
    origin = env.env_origins[VIDEO_TRACK_ENV].detach().cpu().numpy()
    path_shift_y = env.cfg.navigation.field_width * 0.5
    ground_z = origin[2] + VIDEO_GROUND_MARKER_Z

    path_local = env.path_points_local.detach().cpu().numpy()[::2]
    path_world = np.column_stack(
        [
            origin[0] + path_local[:, 0],
            origin[1] + path_local[:, 1] - path_shift_y,
            np.full(path_local.shape[0], ground_z, dtype=np.float32),
        ]
    ).astype(np.float32)

    projected_path, path_visible, _, _ = _project_world_points(path_world, camera_pos, target_pos)
    _draw_projected_polyline(overlay, projected_path, path_visible, (255, 200, 0), 2)

    goal_local = env.goal_local.detach().cpu().numpy()
    goal_center_xy = np.array(
        [origin[0] + goal_local[0], origin[1] + goal_local[1] - path_shift_y],
        dtype=np.float32,
    )
    goal_ground = _make_ground_circle_points(goal_center_xy, env.cfg.navigation.goal_tolerance, ground_z)
    projected_goal, goal_visible, _, _ = _project_world_points(goal_ground, camera_pos, target_pos)
    _draw_projected_polyline(overlay, projected_goal, goal_visible, (0, 220, 0), 3)

    obstacle_local = env.obstacle_positions[VIDEO_TRACK_ENV].detach().cpu().numpy()
    obstacle_radii = env.obstacle_radii[VIDEO_TRACK_ENV].detach().cpu().numpy()
    for idx, radius in enumerate(obstacle_radii):
        if radius <= 0.0:
            continue
        obstacle_center_xy = np.array(
            [origin[0] + obstacle_local[idx, 0], origin[1] + obstacle_local[idx, 1] - path_shift_y],
            dtype=np.float32,
        )
        obstacle_ground = _make_ground_circle_points(obstacle_center_xy, radius, ground_z)
        projected_obstacle, obstacle_visible = _project_world_points(
            obstacle_ground, camera_pos, target_pos
        )[:2]
        _draw_projected_polyline(overlay, projected_obstacle, obstacle_visible, (0, 0, 255), 3)

    camera_scan_points = _camera_scan_world_points(env, scanner, camera_scan)
    projected_scan, scan_visible = _project_world_points(camera_scan_points, camera_pos, target_pos)[:2]
    _draw_projected_points(overlay, projected_scan, scan_visible, (255, 0, 255), 4)

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
    if len(camera_scan_points) > 0:
        cv2.putText(
            overlay,
            "camera scan",
            (24, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
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


def _write_video_frame(env, camera_handle, camera_scan=None, scanner=None):
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
    return _draw_navigation_overlay(frame_bgr, env, camera_pos, target_pos, camera_scan, scanner)


def _configure_oracle_demo(env_cfg):
    nav_cfg = env_cfg.navigation

    env_cfg.env.num_envs = 1
    env_cfg.env.is_train = True
    env_cfg.env.enable_camera_sensors = CREATE_VIDEO
    env_cfg.env.episode_length_s = 180
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.commands.curriculum = False
    env_cfg.commands.trap_time = 30
    nav_cfg.use_terrain_mesh_obstacles = False
    nav_cfg.terminate_on_body_contact = False
    nav_cfg.terminate_on_trap = False
    nav_cfg.terminate_on_collision = True


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
    nav_cfg.depth_camera.enabled = False
    nav_cfg.terminate_on_body_contact = False
    nav_cfg.terminate_on_trap = False
    nav_cfg.terminate_on_collision = True


def _configure_depth_scan_debug_demo(env_cfg):
    _configure_terrain_obstacle_demo(env_cfg)
    env_cfg.env.enable_camera_sensors = True
    env_cfg.navigation.depth_camera.enabled = True


def _configure_depth_camera_demo(env_cfg):
    _configure_depth_scan_debug_demo(env_cfg)


def _create_depth_scan_camera(env, scanner_config):
    depth_cfg = env.cfg.navigation.depth_camera
    camera_props = gymapi.CameraProperties()
    camera_props.width = scanner_config.camera_width
    camera_props.height = scanner_config.camera_height
    camera_props.horizontal_fov = scanner_config.horizontal_fov_deg
    camera_handle = env.gym.create_camera_sensor(env.envs[VIDEO_TRACK_ENV], camera_props)
    if camera_handle == -1:
        raise RuntimeError("Failed to create depth camera sensor.")

    body_handle = env.gym.find_actor_rigid_body_handle(
        env.envs[VIDEO_TRACK_ENV],
        env.actor_handles[VIDEO_TRACK_ENV],
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
        env.envs[VIDEO_TRACK_ENV],
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


def _attached_depth_camera_frame(env):
    root_pos = env.root_states[VIDEO_TRACK_ENV, :3].detach().cpu().numpy()
    root_quat = env.root_states[VIDEO_TRACK_ENV, 3:7].detach().cpu().numpy()

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
        env.envs[VIDEO_TRACK_ENV],
        camera_handle,
        gymapi.IMAGE_DEPTH,
    )
    if depth_image is None or len(depth_image) == 0:
        raise RuntimeError("Depth camera returned an empty image.")
    return np.reshape(depth_image, (scanner_config.camera_height, scanner_config.camera_width))


def _compute_oracle_scan(env, scanner):
    env_id = VIDEO_TRACK_ENV
    return scanner.compute(
        "oracle",
        sensor_xy=env.nav_local_pos[env_id].detach().cpu().numpy(),
        heading=float(env.nav_heading[env_id].item()),
        obstacle_positions=env.obstacle_positions[env_id].detach().cpu().numpy(),
        obstacle_radii=env.obstacle_radii[env_id].detach().cpu().numpy(),
    )


def _compute_camera_scan(env, depth_camera, scanner, scanner_config):
    depth_image = _read_depth_image(env, depth_camera, scanner_config)
    camera_pos, camera_forward, camera_right, camera_up = _attached_depth_camera_frame(env)
    ground_height = float(env.env_origins[VIDEO_TRACK_ENV, 2].item())
    camera_scan = scanner.compute(
        "camera",
        depth_image=depth_image,
        camera_position=camera_pos,
        camera_forward=camera_forward,
        camera_right=camera_right,
        camera_up=camera_up,
        ground_height=ground_height,
    )
    return camera_scan, camera_pos


def _make_depth_nav_observation(env, depth_camera, scanner, scanner_config, translator):
    camera_scan, camera_pos = _compute_camera_scan(env, depth_camera, scanner, scanner_config)
    oracle_nav_obs = env.get_nav_observations()[VIDEO_TRACK_ENV].detach().cpu().numpy()
    translation = translator.translate(base_nav_obs=oracle_nav_obs, scan=camera_scan)
    nav_obs = torch.as_tensor(translation.nav_obs, dtype=torch.float32, device=env.device).unsqueeze(0)
    return nav_obs, translation, camera_scan, camera_pos, oracle_nav_obs


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


def _format_scan_summary(prefix, summary):
    band_front = ",".join(f"{value:.2f}" for value in summary["band_front_min"])
    return (
        f"{prefix}: "
        f"front={summary['front_min']:.2f} "
        f"left={summary['left_min']:.2f} "
        f"right={summary['right_min']:.2f} "
        f"min={summary['overall_min']:.2f}@{summary['overall_min_angle_deg']:.1f}deg "
        f"band_front=[{band_front}]"
    )


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


def _max_episode_steps(env):
    if hasattr(env.max_episode_length, "item"):
        return int(env.max_episode_length.item())
    return int(env.max_episode_length)


def _print_navigation_status(env, step, last_nav_action):
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


def _print_navigation_reset(env, nav_rewards, last_nav_action):
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


def _oracle_mode_message(_env):
    return (
        "oracle navigation mode: using environment-provided obstacle slots; "
        "D435 camera-derived slots are not used"
    )


def _terrain_mode_message(env):
    terrain_seed = getattr(getattr(env, "terrain", None), "obstacle_seed", None)
    return (
        "terrain obstacle mode: static trimesh obstacles are synced into navigation observations; "
        "D435 camera-derived slots are not used | "
        f"seed={terrain_seed}"
    )


def _depth_scan_debug_mode_message(env):
    terrain_seed = getattr(getattr(env, "terrain", None), "obstacle_seed", None)
    return (
        "depth-scan debug mode: camera and oracle scans are compared side by side; "
        "camera-derived slots are not fed into the navigation actor | "
        f"seed={terrain_seed}"
    )


def _depth_camera_mode_message(env):
    terrain_seed = getattr(getattr(env, "terrain", None), "obstacle_seed", None)
    return (
        "depth-camera navigation mode: the navigation actor receives camera-derived "
        "24D obstacle slots and the safety shield uses camera-derived front clearance | "
        f"seed={terrain_seed}"
    )


def _play_policy_navigation_demo(args, configure_demo, experiment_name, mode_message):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    configure_demo(env_cfg)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    video_writer = None
    video_path = None
    camera_handle = None
    if CREATE_VIDEO:
        camera_handle, video_writer, video_path = _create_video_recorder(
            env, experiment_name
        )

    locomotion_policy = _load_jit_policy(env_cfg.navigation.locomotion_policy_path, env.device, "Locomotion")
    recovery_policy = _load_jit_policy(env_cfg.navigation.recovery_policy_path, env.device, "Recovery")
    nav_policy, _ = load_navigation_policy(env_cfg.navigation.nav_policy_path, env.device)

    print(mode_message(env))

    nav_obs = env.get_nav_observations()
    max_steps = _max_episode_steps(env)
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
                _print_navigation_status(env, step, last_nav_action)

            if bool(nav_dones[0].item()):
                _print_navigation_reset(env, nav_rewards, last_nav_action)
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"video saved to: {video_path}")


def _play_oracle(args):
    return _play_policy_navigation_demo(
        args,
        _configure_oracle_demo,
        "roll_robot_r_hierarchical_nav",
        _oracle_mode_message,
    )


def _play_terrain(args):
    return _play_policy_navigation_demo(
        args,
        _configure_terrain_obstacle_demo,
        "roll_robot_r_hierarchical_nav_terrain",
        _terrain_mode_message,
    )


def _play_depth_scan_debug(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    _configure_depth_scan_debug_demo(env_cfg)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    scanner_config = create_depth_scan_config(env_cfg.navigation)
    scanner = NavDepthScanner(scanner_config)
    depth_camera = _create_depth_scan_camera(env, scanner_config)

    video_writer = None
    video_path = None
    video_camera = None
    if CREATE_VIDEO:
        video_camera, video_writer, video_path = _create_video_recorder(
            env, "roll_robot_r_hierarchical_nav_depth_scan_debug"
        )

    locomotion_policy = _load_jit_policy(env_cfg.navigation.locomotion_policy_path, env.device, "Locomotion")
    recovery_policy = _load_jit_policy(env_cfg.navigation.recovery_policy_path, env.device, "Recovery")
    nav_policy, _ = load_navigation_policy(env_cfg.navigation.nav_policy_path, env.device)

    print(_depth_scan_debug_mode_message(env))

    nav_obs = env.get_nav_observations()
    max_steps = _max_episode_steps(env)
    last_camera_scan = None

    try:
        for step in range(max_steps):
            with torch.inference_mode():
                nav_actions = nav_policy.act(nav_obs)
                env.apply_navigation_actions(nav_actions)
                low_level_actions = _compose_low_level_actions(env, locomotion_policy, recovery_policy)
                nav_obs, _, _, nav_dones, _, _ = env.step(low_level_actions)

            if step % DEPTH_SCAN_PRINT_STRIDE == 0:
                camera_scan, camera_pos = _compute_camera_scan(env, depth_camera, scanner, scanner_config)
                oracle_scan = _compute_oracle_scan(env, scanner)
                comparison = compare_depth_scans(scanner, camera_scan, oracle_scan)
                last_camera_scan = camera_scan

                status = env.get_navigation_status(VIDEO_TRACK_ENV)
                print(
                    f"step={step:04d} "
                    f"x={status['local_x']:.2f} "
                    f"y={status['local_y']:.2f} "
                    f"cam_z={camera_pos[2]:.2f} "
                    f"{_format_scan_summary('camera', comparison.camera_summary)} | "
                    f"{_format_scan_summary('oracle', comparison.oracle_summary)} | "
                    f"{format_depth_scan_comparison(comparison)}"
                )

            if (
                video_writer is not None
                and video_camera is not None
                and step % VIDEO_FRAME_STRIDE == 0
            ):
                frame_bgr = _write_video_frame(env, video_camera, last_camera_scan, scanner)
                video_writer.write(frame_bgr)

            if bool(nav_dones[VIDEO_TRACK_ENV].item()):
                reasons = env.get_termination_status(VIDEO_TRACK_ENV)
                print(
                    "episode reset | "
                    f"goal={reasons['goal_reached']} "
                    f"collision={reasons['collision']} "
                    f"out_of_bounds={reasons['out_of_bounds']} "
                    f"body_contact={reasons['body_contact']} "
                    f"trap={reasons['trap']} "
                    f"timeout={reasons['timeout']}"
                )
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"video saved to: {video_path}")


def _play_depth_camera(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    _configure_depth_camera_demo(env_cfg)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    scanner_config = create_depth_scan_config(env_cfg.navigation)
    scanner = NavDepthScanner(scanner_config)
    translator = create_depth_nav_translator(env_cfg.navigation, scanner)
    depth_camera = _create_depth_scan_camera(env, scanner_config)

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

    print(_depth_camera_mode_message(env))

    max_steps = _max_episode_steps(env)
    last_nav_action = torch.zeros(2, device=env.device)
    last_camera_scan = None

    try:
        for step in range(max_steps):
            camera_nav_obs, translation, camera_scan, camera_pos, oracle_nav_obs = _make_depth_nav_observation(
                env,
                depth_camera,
                scanner,
                scanner_config,
                translator,
            )
            oracle_nav_obs_tensor = torch.as_tensor(
                oracle_nav_obs,
                dtype=torch.float32,
                device=env.device,
            ).unsqueeze(0)

            with torch.inference_mode():
                camera_nav_actions = nav_policy.act(camera_nav_obs)
                oracle_nav_actions = nav_policy.act(oracle_nav_obs_tensor)
                last_nav_action = camera_nav_actions[0].detach().clone()
                policy_comparison = compare_depth_policy_inputs(
                    oracle_nav_obs=oracle_nav_obs,
                    camera_nav_obs=translation.nav_obs,
                    oracle_action=oracle_nav_actions[0].detach().cpu().numpy(),
                    camera_action=camera_nav_actions[0].detach().cpu().numpy(),
                    nav_state_dim=8,
                )
                _apply_navigation_actions_with_depth_clearance(
                    env,
                    camera_nav_actions,
                    translation.front_clearance,
                )
                low_level_actions = _compose_low_level_actions(env, locomotion_policy, recovery_policy)
                _, _, nav_rewards, nav_dones, _, _ = env.step(low_level_actions)

            last_camera_scan = camera_scan

            if (
                video_writer is not None
                and video_camera is not None
                and step % VIDEO_FRAME_STRIDE == 0
            ):
                frame_bgr = _write_video_frame(env, video_camera, last_camera_scan, scanner)
                video_writer.write(frame_bgr)

            if step % DEPTH_CAMERA_PRINT_STRIDE == 0:
                status = env.get_navigation_status(VIDEO_TRACK_ENV)
                command = env.nav_command_buffer[VIDEO_TRACK_ENV]
                nav_action = last_nav_action
                print(
                    f"step={step:04d} "
                    f"x={status['local_x']:.2f} "
                    f"y={status['local_y']:.2f} "
                    f"cam_z={camera_pos[2]:.2f} "
                    f"goal_dist={status['goal_distance']:.2f} "
                    f"track_err={status['track_error']:.2f} "
                    f"{_format_depth_status(translation)} "
                    f"{format_depth_policy_comparison(policy_comparison)} "
                    f"blend={status['path_blend']:.2f} "
                    f"speed_scale={status['speed_scale']:.2f} "
                    f"nav=(rudder={nav_action[0].item():.2f}, thrust={nav_action[1].item():.2f}) "
                    f"cmd=(vx={command[0].item():.2f}, yaw={command[2].item():.2f})"
                )

            if bool(nav_dones[VIDEO_TRACK_ENV].item()):
                status = env.get_navigation_status(VIDEO_TRACK_ENV)
                reasons = env.get_termination_status(VIDEO_TRACK_ENV)
                command = env.nav_command_buffer[VIDEO_TRACK_ENV]
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
                    f"{format_depth_policy_comparison(policy_comparison)} "
                    f"blend={status['path_blend']:.2f} "
                    f"speed_scale={status['speed_scale']:.2f} "
                    f"nav=(rudder={nav_action[0].item():.2f}, thrust={nav_action[1].item():.2f}) "
                    f"cmd=(vx={command[0].item():.2f}, yaw={command[2].item():.2f}) "
                    f"reward={nav_rewards[VIDEO_TRACK_ENV].item():.2f}"
                )
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"video saved to: {video_path}")


def play(args):
    nav_mode = getattr(args, "nav_mode", DEFAULT_NAV_MODE)
    if nav_mode == "oracle":
        return _play_oracle(args)
    if nav_mode == "terrain":
        return _play_terrain(args)
    if nav_mode == "depth-scan-debug":
        return _play_depth_scan_debug(args)
    if nav_mode == "depth-camera":
        return _play_depth_camera(args)
    raise ValueError(f"Unsupported hierarchical navigation mode: {nav_mode}")


def parse_hierarchical_nav_args():
    nav_mode = DEFAULT_NAV_MODE
    remaining = [sys.argv[0]]
    idx = 1
    while idx < len(sys.argv):
        item = sys.argv[idx]
        if item == "--nav-mode" or item == "--mode":
            if idx + 1 >= len(sys.argv):
                raise ValueError(f"{item} requires one of: {', '.join(CANONICAL_NAV_MODES)}")
            nav_mode = sys.argv[idx + 1]
            idx += 2
            continue
        if item.startswith("--nav-mode="):
            nav_mode = item.split("=", 1)[1]
            idx += 1
            continue
        if item.startswith("--mode="):
            nav_mode = item.split("=", 1)[1]
            idx += 1
            continue
        remaining.append(item)
        idx += 1

    if nav_mode not in CANONICAL_NAV_MODES:
        raise ValueError(
            f"Unsupported --nav-mode '{nav_mode}'. Supported modes: {', '.join(CANONICAL_NAV_MODES)}"
        )

    original_argv = sys.argv
    try:
        sys.argv = remaining
        args = get_args()
    finally:
        sys.argv = original_argv
    args.nav_mode = nav_mode
    return args


if __name__ == "__main__":
    args = parse_hierarchical_nav_args()
    args.task = "roll_robot_r_hierarchical_nav"
    args.headless = True
    play(args)
