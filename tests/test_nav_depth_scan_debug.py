"""Pure Python checks for depth-scan debug comparison output."""

import unittest

import numpy as np

from legged_gym.utils.nav_depth_debug import (
    compare_depth_policy_inputs,
    compare_depth_scans,
    format_depth_policy_comparison,
    format_depth_scan_comparison,
)
from legged_gym.utils.nav_depth_scan import DepthScanConfig, NavDepthScanner


class NavDepthScanDebugTest(unittest.TestCase):
    def test_comparison_reports_camera_oracle_clearance_differences(self):
        config = DepthScanConfig(
            num_rays=5,
            horizontal_fov_deg=80.0,
            min_depth=0.2,
            max_depth=6.0,
            front_angle_deg=10.0,
            side_min_angle_deg=20.0,
            band_row_ranges=((0.0, 1.0),),
        )
        scanner = NavDepthScanner(config)
        camera_scan = np.full((1, 5), 6.0, dtype=np.float32)
        oracle_scan = np.full((1, 5), 6.0, dtype=np.float32)
        camera_scan[0, 2] = 1.50
        oracle_scan[0, 2] = 1.20
        camera_scan[0, 4] = 3.00
        oracle_scan[0, 4] = 2.50

        comparison = compare_depth_scans(scanner, camera_scan, oracle_scan)

        self.assertAlmostEqual(comparison.front_delta, 0.30, places=5)
        self.assertAlmostEqual(comparison.left_delta, 0.50, places=5)
        self.assertAlmostEqual(comparison.nearest_delta, 0.30, places=5)
        self.assertAlmostEqual(comparison.nearest_angle_delta_deg, 0.0, places=5)

        report = format_depth_scan_comparison(comparison)
        self.assertIn("front_delta=0.30", report)
        self.assertIn("left_delta=0.50", report)
        self.assertIn("nearest_delta=0.30", report)

    def test_synthetic_camera_scan_matches_oracle_front_obstacle(self):
        config = DepthScanConfig(
            num_rays=5,
            horizontal_fov_deg=90.0,
            min_depth=0.2,
            max_depth=6.0,
            camera_width=20,
            camera_height=10,
            percentile=0.0,
            min_points_per_bin=1,
            front_angle_deg=15.0,
            side_min_angle_deg=20.0,
            band_row_ranges=((0.0, 1.0),),
        )
        scanner = NavDepthScanner(config)
        depth_image = np.full((10, 20), 6.0, dtype=np.float32)
        depth_image[:, 8:12] = 1.20

        camera_scan = scanner.compute("camera", depth_image=depth_image)
        oracle_scan = scanner.compute(
            "oracle",
            sensor_xy=np.array([0.0, 0.0], dtype=np.float32),
            heading=0.0,
            obstacle_positions=np.array([[1.65, 0.0]], dtype=np.float32),
            obstacle_radii=np.array([0.45], dtype=np.float32),
        )

        comparison = compare_depth_scans(scanner, camera_scan, oracle_scan)

        self.assertAlmostEqual(comparison.front_delta, 0.0, places=5)
        self.assertAlmostEqual(comparison.nearest_delta, 0.0, places=5)
        self.assertAlmostEqual(comparison.nearest_angle_delta_deg, 0.0, places=5)

    def test_policy_comparison_reports_observation_and_action_drift(self):
        oracle_nav_obs = np.zeros(24, dtype=np.float32)
        camera_nav_obs = np.zeros(24, dtype=np.float32)
        oracle_nav_obs[:8] = np.arange(8, dtype=np.float32)
        camera_nav_obs[:8] = np.arange(8, dtype=np.float32)
        oracle_nav_obs[8:12] = np.array([0.20, 0.00, 0.10, 0.09], dtype=np.float32)
        camera_nav_obs[8:12] = np.array([0.30, -0.20, 0.15, 0.13], dtype=np.float32)

        comparison = compare_depth_policy_inputs(
            oracle_nav_obs=oracle_nav_obs,
            camera_nav_obs=camera_nav_obs,
            oracle_action=np.array([0.10, 0.50], dtype=np.float32),
            camera_action=np.array([-0.20, 0.70], dtype=np.float32),
            nav_state_dim=8,
        )

        self.assertAlmostEqual(comparison.nav_state_max_abs_delta, 0.0, places=5)
        self.assertAlmostEqual(comparison.obstacle_slot_max_abs_delta, 0.20, places=5)
        self.assertAlmostEqual(comparison.action_max_abs_delta, 0.30, places=5)
        self.assertGreater(comparison.obstacle_slot_l2_delta, 0.0)
        self.assertGreater(comparison.action_l2_delta, 0.0)

        report = format_depth_policy_comparison(comparison)
        self.assertIn("obs_slot_max_delta=0.20", report)
        self.assertIn("action_max_delta=0.30", report)


if __name__ == "__main__":
    unittest.main()
