import math

import numpy as np

from legged_gym.utils.nav_depth_observation import DepthNavObservationTranslator
from legged_gym.utils.nav_depth_scan import DepthScanConfig


def _get_cfg_value(cfg, name, default):
    return getattr(cfg, name, default)


def create_depth_scan_config(nav_cfg):
    depth_cfg = nav_cfg.depth_camera
    return DepthScanConfig(
        num_rays=int(_get_cfg_value(depth_cfg, "num_rays", 61)),
        horizontal_fov_deg=float(_get_cfg_value(depth_cfg, "horizontal_fov_deg", 87.0)),
        vertical_fov_deg=float(_get_cfg_value(depth_cfg, "vertical_fov_deg", 58.0)),
        min_depth=float(_get_cfg_value(depth_cfg, "min_depth", 0.28)),
        max_depth=float(_get_cfg_value(depth_cfg, "max_depth", 6.0)),
        camera_width=int(_get_cfg_value(depth_cfg, "width", 848)),
        camera_height=int(_get_cfg_value(depth_cfg, "height", 480)),
        percentile=float(_get_cfg_value(depth_cfg, "percentile", 10.0)),
        min_points_per_bin=int(_get_cfg_value(depth_cfg, "min_points_per_bin", 8)),
        ground_filter_min_height=float(_get_cfg_value(depth_cfg, "ground_filter_min_height", 0.08)),
        ground_filter_max_height=float(_get_cfg_value(depth_cfg, "ground_filter_max_height", 2.0)),
        front_angle_deg=float(_get_cfg_value(depth_cfg, "front_angle_deg", 15.0)),
        side_min_angle_deg=float(_get_cfg_value(depth_cfg, "side_min_angle_deg", 20.0)),
        band_row_ranges=tuple(_get_cfg_value(depth_cfg, "band_row_ranges", DepthScanConfig().band_row_ranges)),
    )


def create_depth_nav_translator(nav_cfg, scanner):
    radius_range = getattr(nav_cfg, "obstacle_radius_range", [0.45, 0.85])
    depth_cfg = nav_cfg.depth_camera
    default_radius = 0.5 * (float(radius_range[0]) + float(radius_range[1]))
    detection_max_distance = min(float(scanner.config.max_depth) * 0.98, float(nav_cfg.observation_radius))

    return DepthNavObservationTranslator(
        scanner=scanner,
        observation_radius=nav_cfg.observation_radius,
        robot_radius=nav_cfg.robot_radius,
        num_obstacles=nav_cfg.num_nearest_obstacles,
        nav_state_dim=8,
        default_obstacle_radius=default_radius,
        min_obstacle_radius=radius_range[0],
        max_obstacle_radius=radius_range[1],
        detection_max_distance=detection_max_distance,
        front_angle_deg=scanner.config.front_angle_deg,
        cluster_percentile=float(_get_cfg_value(depth_cfg, "cluster_percentile", 15.0)),
        max_cluster_gap_rays=int(_get_cfg_value(depth_cfg, "max_cluster_gap_rays", 1)),
        min_cluster_rays=int(_get_cfg_value(depth_cfg, "min_cluster_rays", 2)),
    )


def depth_camera_mount_position(nav_cfg):
    return np.asarray(nav_cfg.depth_camera.local_position, dtype=np.float32)


def depth_camera_mount_euler_deg(nav_cfg):
    depth_cfg = nav_cfg.depth_camera
    return (
        float(_get_cfg_value(depth_cfg, "roll_deg", 0.0)),
        float(_get_cfg_value(depth_cfg, "pitch_deg", 0.0)),
        float(_get_cfg_value(depth_cfg, "yaw_deg", 0.0)),
    )


def depth_camera_local_frame(nav_cfg):
    roll_deg, pitch_deg, yaw_deg = depth_camera_mount_euler_deg(nav_cfg)
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float32,
    )
    rot_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float32,
    )
    rot_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rotation = rot_z @ rot_y @ rot_x

    local_forward = rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    local_right = rotation @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
    local_up = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (
        local_forward.astype(np.float32),
        local_right.astype(np.float32),
        local_up.astype(np.float32),
    )


def depth_camera_mount_quat(gymapi, nav_cfg):
    roll_deg, pitch_deg, yaw_deg = depth_camera_mount_euler_deg(nav_cfg)
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    w = cy * cp * cr + sy * sp * sr
    return gymapi.Quat(float(x), float(y), float(z), float(w))
