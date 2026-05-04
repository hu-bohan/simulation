import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DepthScanConfig:
    num_rays: int = 61
    horizontal_fov_deg: float = 87.0
    vertical_fov_deg: float = 58.0
    min_depth: float = 0.28
    max_depth: float = 6.0
    camera_width: int = 640
    camera_height: int = 480
    percentile: float = 10.0
    min_points_per_bin: int = 8
    front_angle_deg: float = 15.0
    side_min_angle_deg: float = 20.0
    band_row_ranges: tuple = (
        (0.58, 0.88),  # low image band, usually ground-level obstacles
        (0.36, 0.66),  # mid image band
        (0.14, 0.44),  # high image band
    )

    @property
    def num_bands(self):
        return len(self.band_row_ranges)

    @property
    def ray_angles(self):
        half_fov = math.radians(self.horizontal_fov_deg) * 0.5
        return np.linspace(-half_fov, half_fov, self.num_rays, dtype=np.float32)


class NavDepthScanner:
    """Converts either known obstacle geometry or depth images into polar scans.

    Output shape is [num_bands, num_rays] for one frame. Distances are meters in
    the sensor/body horizontal plane. Positive scan angles point to the robot's
    left, matching the navigation body-y convention used by hierarchical_nav.
    """

    def __init__(self, config=None):
        self.config = config or DepthScanConfig()

    def compute(self, source, **kwargs):
        if source == "oracle":
            return self.compute_oracle(**kwargs)
        if source == "camera":
            return self.compute_camera(**kwargs)
        raise ValueError("source must be 'oracle' or 'camera'")

    def compute_oracle(self, sensor_xy, heading, obstacle_positions, obstacle_radii):
        sensor_xy = np.asarray(sensor_xy, dtype=np.float32)
        obstacle_positions = np.asarray(obstacle_positions, dtype=np.float32)
        obstacle_radii = np.asarray(obstacle_radii, dtype=np.float32)

        if sensor_xy.ndim != 1 or sensor_xy.shape[0] != 2:
            raise ValueError("sensor_xy must be a 2D position")
        if obstacle_positions.ndim != 2 or obstacle_positions.shape[1] != 2:
            raise ValueError("obstacle_positions must have shape [num_obstacles, 2]")
        if obstacle_radii.ndim != 1 or obstacle_radii.shape[0] != obstacle_positions.shape[0]:
            raise ValueError("obstacle_radii must have shape [num_obstacles]")

        cfg = self.config
        scan_1d = np.full((cfg.num_rays,), cfg.max_depth, dtype=np.float32)
        heading = float(heading)

        for ray_idx, relative_angle in enumerate(cfg.ray_angles):
            ray_heading = heading + float(relative_angle)
            ray_dir = np.array([math.cos(ray_heading), math.sin(ray_heading)], dtype=np.float32)
            nearest = cfg.max_depth

            for center, radius in zip(obstacle_positions, obstacle_radii):
                if radius <= 0.0:
                    continue
                hit_distance = self._ray_circle_intersection(sensor_xy, ray_dir, center, float(radius))
                if hit_distance is not None and cfg.min_depth <= hit_distance < nearest:
                    nearest = hit_distance

            scan_1d[ray_idx] = nearest

        return np.repeat(scan_1d[None, :], cfg.num_bands, axis=0)

    def compute_camera(self, depth_image):
        depth = self._prepare_depth_image(depth_image)
        cfg = self.config

        height, width = depth.shape
        column_angles = self._column_angles(width)
        ray_edges = self._ray_edges()
        scan = np.full((cfg.num_bands, cfg.num_rays), cfg.max_depth, dtype=np.float32)

        valid_depth = np.isfinite(depth) & (depth >= cfg.min_depth) & (depth <= cfg.max_depth)
        depth = np.where(valid_depth, depth, np.nan)

        for band_idx, (row_start_fraction, row_end_fraction) in enumerate(cfg.band_row_ranges):
            row_start = int(np.clip(round(row_start_fraction * height), 0, height - 1))
            row_end = int(np.clip(round(row_end_fraction * height), row_start + 1, height))
            band_depth = depth[row_start:row_end]

            for ray_idx in range(cfg.num_rays):
                col_mask = (column_angles >= ray_edges[ray_idx]) & (column_angles < ray_edges[ray_idx + 1])
                if not np.any(col_mask):
                    continue

                values = band_depth[:, col_mask]
                values = values[np.isfinite(values)]
                if values.size < cfg.min_points_per_bin:
                    continue
                scan[band_idx, ray_idx] = float(np.percentile(values, cfg.percentile))

        return scan

    def summarize(self, scan):
        scan = np.asarray(scan, dtype=np.float32)
        if scan.ndim != 2 or scan.shape[1] != self.config.num_rays:
            raise ValueError("scan must have shape [num_bands, num_rays]")

        cfg = self.config
        angles_deg = np.degrees(cfg.ray_angles)
        min_per_ray = np.min(scan, axis=0)

        front_mask = np.abs(angles_deg) <= cfg.front_angle_deg
        left_mask = angles_deg >= cfg.side_min_angle_deg
        right_mask = angles_deg <= -cfg.side_min_angle_deg

        min_idx = int(np.argmin(min_per_ray))
        band_front_min = [self._masked_min(scan[band_idx], front_mask) for band_idx in range(scan.shape[0])]
        band_overall_min = [float(np.min(scan[band_idx])) for band_idx in range(scan.shape[0])]
        return {
            "front_min": self._masked_min(min_per_ray, front_mask),
            "left_min": self._masked_min(min_per_ray, left_mask),
            "right_min": self._masked_min(min_per_ray, right_mask),
            "overall_min": float(min_per_ray[min_idx]),
            "overall_min_angle_deg": float(angles_deg[min_idx]),
            "band_front_min": band_front_min,
            "band_overall_min": band_overall_min,
        }

    def _prepare_depth_image(self, depth_image):
        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.ndim != 2:
            depth = depth.reshape(self.config.camera_height, self.config.camera_width)

        finite = depth[np.isfinite(depth)]
        if finite.size > 0 and np.mean(finite < 0.0) > 0.5:
            depth = -depth
        return depth

    def _column_angles(self, width):
        half_fov = math.radians(self.config.horizontal_fov_deg) * 0.5
        focal_x = width / (2.0 * math.tan(half_fov))
        center_x = (width - 1) * 0.5
        pixel_x = np.arange(width, dtype=np.float32)
        return np.arctan((pixel_x - center_x) / focal_x)

    def _ray_edges(self):
        angles = self.config.ray_angles.astype(np.float32)
        if len(angles) == 1:
            half_width = math.radians(self.config.horizontal_fov_deg) * 0.5
            return np.array([-half_width, half_width], dtype=np.float32)

        mids = 0.5 * (angles[:-1] + angles[1:])
        first = angles[0] - (mids[0] - angles[0])
        last = angles[-1] + (angles[-1] - mids[-1])
        return np.concatenate([[first], mids, [last]]).astype(np.float32)

    @staticmethod
    def _ray_circle_intersection(origin, direction, center, radius):
        center_delta = center - origin
        projected = float(np.dot(center_delta, direction))
        if projected <= 0.0:
            return None

        closest_sq = float(np.dot(center_delta, center_delta) - projected * projected)
        radius_sq = radius * radius
        if closest_sq > radius_sq:
            return None

        offset = math.sqrt(max(radius_sq - closest_sq, 0.0))
        hit_distance = projected - offset
        if hit_distance < 0.0:
            hit_distance = projected + offset
        return hit_distance if hit_distance >= 0.0 else None

    @staticmethod
    def _masked_min(values, mask):
        if not np.any(mask):
            return float("nan")
        return float(np.min(values[mask]))
