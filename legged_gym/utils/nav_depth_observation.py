import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DepthObstacleEstimate:
    body_x: float
    body_y: float
    clearance: float
    radius: float
    surface_distance: float
    angle: float
    ray_count: int


@dataclass
class DepthNavTranslation:
    nav_obs: np.ndarray
    obstacle_features: np.ndarray
    front_clearance: float
    obstacle_count: int
    ray_depth: np.ndarray
    obstacles: list


class DepthNavObservationTranslator:
    """Translate a depth camera scan into the existing navigation policy input.

    The trained navigation actor expects the last part of its observation to be
    four nearest obstacle slots, each encoded as body-frame x/y, clearance, and
    radius. This adapter estimates those slots from a depth image while keeping
    the observation dimensionality unchanged.
    """

    def __init__(
        self,
        scanner,
        observation_radius,
        robot_radius,
        num_obstacles=4,
        nav_state_dim=8,
        default_obstacle_radius=0.65,
        min_obstacle_radius=0.45,
        max_obstacle_radius=0.85,
        detection_max_distance=None,
        front_angle_deg=15.0,
        cluster_percentile=15.0,
        max_cluster_gap_rays=1,
        min_cluster_rays=2,
    ):
        self.scanner = scanner
        self.config = scanner.config
        self.observation_radius = float(observation_radius)
        self.robot_radius = float(robot_radius)
        self.num_obstacles = int(num_obstacles)
        self.nav_state_dim = int(nav_state_dim)
        self.default_obstacle_radius = float(default_obstacle_radius)
        self.min_obstacle_radius = float(min_obstacle_radius)
        self.max_obstacle_radius = float(max_obstacle_radius)
        self.detection_max_distance = (
            float(detection_max_distance)
            if detection_max_distance is not None
            else float(self.config.max_depth) * 0.98
        )
        self.front_angle_deg = float(front_angle_deg)
        self.cluster_percentile = float(cluster_percentile)
        self.max_cluster_gap_rays = int(max_cluster_gap_rays)
        self.min_cluster_rays = int(min_cluster_rays)

    def translate(
        self,
        base_nav_obs,
        depth_image=None,
        scan=None,
        camera_position=None,
        camera_forward=None,
        camera_right=None,
        camera_up=None,
        ground_height=None,
    ):
        if scan is None:
            if depth_image is None:
                raise ValueError("Either depth_image or scan must be provided.")
            scan = self.scanner.compute(
                "camera",
                depth_image=depth_image,
                camera_position=camera_position,
                camera_forward=camera_forward,
                camera_right=camera_right,
                camera_up=camera_up,
                ground_height=ground_height,
            )

        base_nav_obs = np.asarray(base_nav_obs, dtype=np.float32).copy()
        expected_dim = self.nav_state_dim + self.num_obstacles * 4
        if base_nav_obs.ndim != 1 or base_nav_obs.shape[0] < expected_dim:
            raise ValueError(
                f"base_nav_obs must be a flat array with at least {expected_dim} values."
            )

        ray_depth = self._collapse_scan(scan)
        obstacles = self._extract_obstacles(ray_depth)
        features = self._obstacles_to_features(obstacles)
        front_clearance = self._compute_front_clearance(ray_depth)

        start = self.nav_state_dim
        end = start + self.num_obstacles * 4
        base_nav_obs[start:end] = features.reshape(-1)

        return DepthNavTranslation(
            nav_obs=base_nav_obs,
            obstacle_features=features,
            front_clearance=front_clearance,
            obstacle_count=len(obstacles),
            ray_depth=ray_depth,
            obstacles=obstacles,
        )

    def _collapse_scan(self, scan):
        scan = np.asarray(scan, dtype=np.float32)
        if scan.ndim != 2 or scan.shape[1] != self.config.num_rays:
            raise ValueError("scan must have shape [num_bands, num_rays].")

        finite_scan = np.where(np.isfinite(scan), scan, self.config.max_depth)
        ray_depth = np.min(finite_scan, axis=0)
        ray_depth = np.where(np.isfinite(ray_depth), ray_depth, self.config.max_depth)
        return np.clip(ray_depth, self.config.min_depth, self.config.max_depth).astype(np.float32)

    def _extract_obstacles(self, ray_depth):
        valid = (
            np.isfinite(ray_depth)
            & (ray_depth >= self.config.min_depth)
            & (ray_depth <= self.detection_max_distance)
        )
        valid = self._fill_small_gaps(valid)

        clusters = self._clusters_from_mask(valid)
        obstacles = []
        for cluster in clusters:
            if len(cluster) < self.min_cluster_rays:
                continue
            obstacle = self._obstacle_from_cluster(cluster, ray_depth)
            if obstacle is not None:
                obstacles.append(obstacle)

        obstacles.sort(key=lambda item: item.clearance)
        return obstacles[: self.num_obstacles]

    def _fill_small_gaps(self, mask):
        if self.max_cluster_gap_rays <= 0:
            return mask

        filled = mask.copy()
        idx = 0
        while idx < len(mask):
            if mask[idx]:
                idx += 1
                continue

            start = idx
            while idx < len(mask) and not mask[idx]:
                idx += 1
            end = idx
            gap_len = end - start
            has_left = start > 0 and mask[start - 1]
            has_right = end < len(mask) and mask[end]
            if has_left and has_right and gap_len <= self.max_cluster_gap_rays:
                filled[start:end] = True

        return filled

    @staticmethod
    def _clusters_from_mask(mask):
        clusters = []
        idx = 0
        while idx < len(mask):
            if not mask[idx]:
                idx += 1
                continue

            start = idx
            while idx < len(mask) and mask[idx]:
                idx += 1
            clusters.append(np.arange(start, idx, dtype=np.int64))

        return clusters

    def _obstacle_from_cluster(self, cluster, ray_depth):
        cluster_depth = ray_depth[cluster]
        finite = np.isfinite(cluster_depth)
        if not np.any(finite):
            return None

        valid_depth = cluster_depth[finite]
        surface_distance = float(np.percentile(valid_depth, self.cluster_percentile))
        if surface_distance < self.config.min_depth or surface_distance > self.detection_max_distance:
            return None

        angles = self.config.ray_angles[cluster][finite]
        weights = 1.0 / np.maximum(valid_depth, 1e-3)
        angle = float(np.average(angles, weights=weights))

        ray_step = self._ray_step()
        angular_width = float(max(np.max(angles) - np.min(angles) + ray_step, ray_step))
        radius_from_width = surface_distance * math.sin(max(angular_width, 0.0) * 0.5)
        if not np.isfinite(radius_from_width) or radius_from_width < self.min_obstacle_radius * 0.5:
            radius = self.default_obstacle_radius
        else:
            radius = radius_from_width
        radius = float(np.clip(radius, self.min_obstacle_radius, self.max_obstacle_radius))

        center_distance = surface_distance + radius
        body_x = center_distance * math.cos(angle)
        body_y = center_distance * math.sin(angle)
        clearance = surface_distance - self.robot_radius

        return DepthObstacleEstimate(
            body_x=body_x,
            body_y=body_y,
            clearance=clearance,
            radius=radius,
            surface_distance=surface_distance,
            angle=angle,
            ray_count=int(len(cluster)),
        )

    def _obstacles_to_features(self, obstacles):
        features = np.zeros((self.num_obstacles, 4), dtype=np.float32)
        far_clearance = self.config.max_depth - self.robot_radius

        for slot in range(self.num_obstacles):
            if slot < len(obstacles):
                obstacle = obstacles[slot]
                body_x = obstacle.body_x
                body_y = obstacle.body_y
                clearance = obstacle.clearance
                radius = obstacle.radius
            else:
                body_x = self.config.max_depth
                body_y = 0.0
                clearance = far_clearance
                radius = 0.0

            features[slot, 0] = np.clip(body_x / self.observation_radius, -1.0, 1.0)
            features[slot, 1] = np.clip(body_y / self.observation_radius, -1.0, 1.0)
            features[slot, 2] = np.clip(clearance / self.observation_radius, -1.0, 1.0)
            features[slot, 3] = np.clip(radius / 5.0, 0.0, 1.0)

        return features

    def _compute_front_clearance(self, ray_depth):
        angles_deg = np.degrees(self.config.ray_angles)
        front_mask = np.abs(angles_deg) <= self.front_angle_deg
        if not np.any(front_mask):
            return float(self.config.max_depth - self.robot_radius)

        front_depth = ray_depth[front_mask]
        front_depth = front_depth[np.isfinite(front_depth)]
        if front_depth.size == 0:
            return float(self.config.max_depth - self.robot_radius)

        surface_distance = float(np.min(front_depth))
        return surface_distance - self.robot_radius

    def _ray_step(self):
        angles = self.config.ray_angles
        if len(angles) < 2:
            return math.radians(self.config.horizontal_fov_deg)
        return float(np.median(np.diff(angles)))
