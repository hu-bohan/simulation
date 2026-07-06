from dataclasses import dataclass

import numpy as np


@dataclass
class DepthScanComparison:
    camera_summary: dict
    oracle_summary: dict
    front_delta: float
    left_delta: float
    right_delta: float
    nearest_delta: float
    nearest_angle_delta_deg: float


@dataclass
class DepthPolicyComparison:
    nav_state_max_abs_delta: float
    obstacle_slot_l2_delta: float
    obstacle_slot_max_abs_delta: float
    action_l2_delta: float
    action_max_abs_delta: float


def compare_depth_scans(scanner, camera_scan, oracle_scan):
    camera_summary = scanner.summarize(camera_scan)
    oracle_summary = scanner.summarize(oracle_scan)

    return DepthScanComparison(
        camera_summary=camera_summary,
        oracle_summary=oracle_summary,
        front_delta=camera_summary["front_min"] - oracle_summary["front_min"],
        left_delta=camera_summary["left_min"] - oracle_summary["left_min"],
        right_delta=camera_summary["right_min"] - oracle_summary["right_min"],
        nearest_delta=camera_summary["overall_min"] - oracle_summary["overall_min"],
        nearest_angle_delta_deg=(
            camera_summary["overall_min_angle_deg"] - oracle_summary["overall_min_angle_deg"]
        ),
    )


def compare_depth_policy_inputs(
    oracle_nav_obs,
    camera_nav_obs,
    oracle_action,
    camera_action,
    nav_state_dim=8,
):
    oracle_nav_obs = np.asarray(oracle_nav_obs, dtype=np.float32).reshape(-1)
    camera_nav_obs = np.asarray(camera_nav_obs, dtype=np.float32).reshape(-1)
    oracle_action = np.asarray(oracle_action, dtype=np.float32).reshape(-1)
    camera_action = np.asarray(camera_action, dtype=np.float32).reshape(-1)

    if oracle_nav_obs.shape != camera_nav_obs.shape:
        raise ValueError("oracle_nav_obs and camera_nav_obs must have the same shape.")
    if oracle_action.shape != camera_action.shape:
        raise ValueError("oracle_action and camera_action must have the same shape.")
    if oracle_nav_obs.shape[0] < nav_state_dim:
        raise ValueError("nav_state_dim cannot exceed observation length.")

    nav_state_delta = camera_nav_obs[:nav_state_dim] - oracle_nav_obs[:nav_state_dim]
    obstacle_slot_delta = camera_nav_obs[nav_state_dim:] - oracle_nav_obs[nav_state_dim:]
    action_delta = camera_action - oracle_action

    return DepthPolicyComparison(
        nav_state_max_abs_delta=_max_abs(nav_state_delta),
        obstacle_slot_l2_delta=float(np.linalg.norm(obstacle_slot_delta)),
        obstacle_slot_max_abs_delta=_max_abs(obstacle_slot_delta),
        action_l2_delta=float(np.linalg.norm(action_delta)),
        action_max_abs_delta=_max_abs(action_delta),
    )


def format_depth_scan_comparison(comparison):
    return (
        f"front_delta={comparison.front_delta:.2f} "
        f"left_delta={comparison.left_delta:.2f} "
        f"right_delta={comparison.right_delta:.2f} "
        f"nearest_delta={comparison.nearest_delta:.2f} "
        f"nearest_angle_delta={comparison.nearest_angle_delta_deg:.1f}deg"
    )


def format_depth_policy_comparison(comparison):
    return (
        f"nav_state_max_delta={comparison.nav_state_max_abs_delta:.2f} "
        f"obs_slot_l2_delta={comparison.obstacle_slot_l2_delta:.2f} "
        f"obs_slot_max_delta={comparison.obstacle_slot_max_abs_delta:.2f} "
        f"action_l2_delta={comparison.action_l2_delta:.2f} "
        f"action_max_delta={comparison.action_max_abs_delta:.2f}"
    )


def _max_abs(values):
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))
