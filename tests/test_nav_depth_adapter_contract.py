"""Pure Python contract checks for the D435 navigation adapter.

Contract under test:
- Input scan/depth distances are meters.
- Scan angles are robot/body-frame horizontal angles; positive is body-left,
  negative is body-right.
- Camera-derived obstacles fill the existing 24D navigation observation slots:
  the first 8 values remain navigation state, then four obstacle slots follow.
- Obstacle slots are ordered nearest-clearance first and encoded as normalized
  body-frame x, body-frame y, clearance, and radius.
- Clearance is surface distance minus robot radius; radius is normalized by the
  same convention as the environment policy input, radius / 5.0.
- Missing or far obstacles fill deterministic fallback slots at max depth with
  zero radius.
"""

import math
import os
import unittest

import numpy as np
import torch

from legged_gym.utils.nav_policy_loader import load_navigation_policy
from legged_gym.utils.nav_depth_observation import DepthNavObservationTranslator
from legged_gym.utils.nav_depth_scan import DepthScanConfig, NavDepthScanner


class NavDepthAdapterContractTest(unittest.TestCase):
    def make_translator(self):
        config = DepthScanConfig(
            num_rays=9,
            horizontal_fov_deg=90.0,
            min_depth=0.2,
            max_depth=6.0,
            front_angle_deg=15.0,
            side_min_angle_deg=20.0,
        )
        scanner = NavDepthScanner(config)
        translator = DepthNavObservationTranslator(
            scanner=scanner,
            observation_radius=6.0,
            robot_radius=0.45,
            num_obstacles=4,
            nav_state_dim=8,
            default_obstacle_radius=0.65,
            min_obstacle_radius=0.45,
            max_obstacle_radius=0.85,
            detection_max_distance=5.5,
            front_angle_deg=15.0,
            cluster_percentile=15.0,
            max_cluster_gap_rays=0,
            min_cluster_rays=1,
        )
        return scanner, translator

    def test_known_scan_translates_to_ordered_normalized_obstacle_slots(self):
        scanner, translator = self.make_translator()
        scan = np.full((scanner.config.num_bands, scanner.config.num_rays), 6.0, dtype=np.float32)
        center_ray = scanner.config.num_rays // 2
        left_ray = center_ray + 2
        right_ray = center_ray - 2

        scan[:, center_ray] = 1.20
        scan[:, right_ray] = 2.00
        scan[:, left_ray] = 3.00

        translation = translator.translate(np.zeros(24, dtype=np.float32), scan=scan)

        self.assertEqual(translation.obstacle_count, 3)
        self.assertAlmostEqual(translation.front_clearance, 0.75, places=5)
        self.assertAlmostEqual(translation.obstacles[0].surface_distance, 1.20, places=5)
        self.assertLess(translation.obstacles[1].body_y, 0.0)
        self.assertGreater(translation.obstacles[2].body_y, 0.0)

        features = translation.obstacle_features
        self.assertAlmostEqual(features[0, 2], 0.75 / 6.0, places=5)
        self.assertAlmostEqual(features[0, 3], 0.65 / 5.0, places=5)
        self.assertLess(features[1, 1], 0.0)
        self.assertGreater(features[2, 1], 0.0)

    def test_missing_or_far_obstacles_use_deterministic_fallback_slots(self):
        scanner, translator = self.make_translator()
        scan = np.full((scanner.config.num_bands, scanner.config.num_rays), 6.0, dtype=np.float32)

        translation = translator.translate(np.arange(24, dtype=np.float32), scan=scan)

        self.assertEqual(translation.obstacle_count, 0)
        self.assertAlmostEqual(translation.front_clearance, 5.55, places=5)
        expected_slot = np.array([1.0, 0.0, 5.55 / 6.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(translation.obstacle_features, np.tile(expected_slot, (4, 1)))
        np.testing.assert_allclose(translation.nav_obs[:8], np.arange(8, dtype=np.float32))

    def test_oracle_scan_uses_positive_angles_for_body_left(self):
        config = DepthScanConfig(
            num_rays=9,
            horizontal_fov_deg=90.0,
            min_depth=0.2,
            max_depth=6.0,
        )
        scanner = NavDepthScanner(config)
        obstacle_positions = np.array([[2.0, 1.0], [2.0, -1.0]], dtype=np.float32)
        obstacle_radii = np.array([0.5, 0.5], dtype=np.float32)

        scan = scanner.compute(
            "oracle",
            sensor_xy=np.array([0.0, 0.0], dtype=np.float32),
            heading=0.0,
            obstacle_positions=obstacle_positions,
            obstacle_radii=obstacle_radii,
        )

        min_per_ray = np.min(scan, axis=0)
        left_ray = int(np.argmin(np.abs(config.ray_angles - math.atan2(1.0, 2.0))))
        right_ray = int(np.argmin(np.abs(config.ray_angles - math.atan2(-1.0, 2.0))))
        self.assertLess(min_per_ray[left_ray], config.max_depth)
        self.assertLess(min_per_ray[right_ray], config.max_depth)
        self.assertGreater(config.ray_angles[left_ray], 0.0)
        self.assertLess(config.ray_angles[right_ray], 0.0)

    def test_synthetic_d435_depth_image_feeds_trained_navigation_actor_shape(self):
        policy_path = "logs/nav/td3_ship_best_actor.pt"
        if not os.path.exists(policy_path):
            self.skipTest(f"Navigation policy checkpoint is not available: {policy_path}")

        config = DepthScanConfig(
            num_rays=9,
            horizontal_fov_deg=90.0,
            min_depth=0.2,
            max_depth=6.0,
            camera_width=36,
            camera_height=18,
            percentile=0.0,
            min_points_per_bin=1,
            front_angle_deg=15.0,
            side_min_angle_deg=20.0,
            band_row_ranges=((0.0, 1.0),),
        )
        scanner = NavDepthScanner(config)
        translator = DepthNavObservationTranslator(
            scanner=scanner,
            observation_radius=6.0,
            robot_radius=0.45,
            num_obstacles=4,
            nav_state_dim=8,
            default_obstacle_radius=0.65,
            min_obstacle_radius=0.45,
            max_obstacle_radius=0.85,
            detection_max_distance=5.5,
            front_angle_deg=15.0,
            cluster_percentile=15.0,
            max_cluster_gap_rays=0,
            min_cluster_rays=1,
        )
        depth_image = np.full((config.camera_height, config.camera_width), 6.0, dtype=np.float32)
        depth_image[:, 16:20] = 1.20
        base_nav_obs = np.zeros(24, dtype=np.float32)
        base_nav_obs[:8] = np.array([0.1, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.9], dtype=np.float32)

        translation = translator.translate(base_nav_obs=base_nav_obs, depth_image=depth_image)
        nav_policy, _ = load_navigation_policy(policy_path, "cpu")
        action = nav_policy.act(torch.as_tensor(translation.nav_obs, dtype=torch.float32).unsqueeze(0))

        self.assertEqual(translation.nav_obs.shape, (24,))
        np.testing.assert_allclose(translation.nav_obs[:8], base_nav_obs[:8])
        self.assertEqual(action.shape, (1, 2))
        self.assertTrue(torch.isfinite(action).all().item())
        self.assertLessEqual(float(action.abs().max().item()), 1.0)


if __name__ == "__main__":
    unittest.main()
