# Hierarchical Navigation Depth Camera Plan

This document records the agreed next-step plan for the hierarchical navigation
and Intel RealSense D435 integration work.

## Decisions

- The existing trained policies are fixed inputs for this stage.
- The low-level locomotion/recovery policy bridge is assumed to work.
- The upper-level navigation policy input remains the existing 24D vector.
- D435 integration should convert depth camera data into the existing obstacle
  slots instead of changing the navigation network.
- The next immediate deliverable is a plan and cleanup direction, not code.
- The final primary entry point should be one feature-complete script.
- Old test/demo scripts should be kept, but moved into a separate legacy folder
  for reference.
- D435 parameters and camera mount pose should live in
  `hierarchical_nav_cfg.py`.
- Camera mount sweeping is not in scope for the first cleanup pass.

## Target End State

There should be one first-class hierarchical navigation demo entry point that
can run the major modes through configuration or command-line options:

- oracle obstacle navigation baseline
- terrain obstacle navigation baseline
- depth scan debug mode
- depth camera navigation mode
- optional collision probe/debug mode

The legacy scripts should remain available for recovery and comparison, but
they should no longer be the normal place to add new behavior.

Suggested legacy folder:

`legged_gym/scripts/legacy/`

Suggested first-class entry point:

`legged_gym/scripts/play_hierarchical_nav.py`

Alternative if keeping the old baseline script untouched feels safer:

`legged_gym/scripts/play_hierarchical_nav_demo.py`

The preferred direction is to make `play_hierarchical_nav.py` the full entry
point, because it is already the natural baseline name.

## Current Script Inventory

Keep behavior, but consolidate responsibilities:

- `play_hierarchical_nav.py`
  - Current oracle/baseline hierarchical navigation demo.
  - Should become the first-class entry point.
- `play_hierarchical_nav_terrain.py`
  - Current terrain/trimesh obstacle demo.
  - Should become a mode in the first-class entry point.
- `play_hierarchical_nav_depth_scan_debug.py`
  - Current camera scan vs oracle scan debug script.
  - Should become a debug mode in the first-class entry point.
- `play_hierarchical_nav_depth_camera.py`
  - Current depth camera to navigation actor script.
  - Should become the depth-camera mode in the first-class entry point.
- `play_hierarchical_nav_terrain_collision_probe.py`
  - Current fixed-command collision probe.
  - Can become a debug/probe mode or stay as legacy if rarely used.
- `play_navigation_test.py`
  - Older test script using `roll_robot_r_history_imitate`.
  - Move to legacy unless the user identifies an active use.
- `play_wjk_pre_demo.py` and `play_wjk_pre_demo_old.py`
  - Older demo scripts.
  - Move to legacy.

Archive status:

- The first-class entry point is `legged_gym/scripts/play_hierarchical_nav.py`.
- Historical scripts are archived under `legged_gym/scripts/legacy/`.
- Operational commands and validation order are documented in
  `docs/runbooks/hierarchical_nav_d435_runbook.md`.

## Proposed Entry Point Modes

The unified entry point should support these modes:

- `oracle`
  - Uses environment-provided obstacle slots.
  - No depth camera input to policy.
- `terrain`
  - Uses trimesh obstacles synced into navigation observations.
  - No depth camera input to policy.
- `depth-scan-debug`
  - Creates a simulated D435 camera.
  - Computes camera-derived scan and oracle scan side by side.
  - Reports/visualizes differences.
  - Does not have to feed camera slots into policy.
- `depth-camera`
  - Creates a simulated D435 camera.
  - Converts depth to obstacle slots.
  - Feeds camera-derived 24D navigation observation to the policy.
  - Uses camera-derived front clearance for the safety shield.
- `collision-probe`
  - Optional fixed-command mode for checking physical obstacle collision and
    reset behavior.

## D435 Configuration Placement

Add a camera/depth adapter config section under:

`legged_gym/envs/roll_robot_r_imitate/env_cfg/hierarchical_nav_cfg.py`

Recommended shape:

```python
class navigation:
    ...

    class depth_camera:
        enabled = False
        mount_body = "base_link"
        local_position = [0.23, 0.0, 0.16]
        roll_deg = 0.0
        pitch_deg = 0.0
        yaw_deg = 0.0

        width = 848
        height = 480
        horizontal_fov_deg = 87.0
        vertical_fov_deg = 58.0
        min_depth = 0.28
        max_depth = 6.0

        num_rays = 61
        front_angle_deg = 15.0
        side_min_angle_deg = 20.0
        percentile = 10.0
        min_points_per_bin = 8
        ground_filter_min_height = 0.08
        ground_filter_max_height = 2.0
        band_row_ranges = (
            (0.58, 0.88),
            (0.36, 0.66),
            (0.14, 0.44),
        )

        cluster_percentile = 15.0
        max_cluster_gap_rays = 1
        min_cluster_rays = 2
```

The exact names can be adjusted to fit local style, but the core requirement is
that mount pose and D435 stream parameters are no longer hard-coded in play
scripts.

## Verification Targets

The first real validation path should prove the adapter layer before judging
policy behavior.

Required sim-side checks:

- Fixed known obstacle layouts.
- Camera-derived obstacle positions compared with oracle obstacle positions.
- Camera-derived 24D navigation observation compared with oracle 24D
  navigation observation.
- Video overlay showing:
  - desired path
  - goal
  - oracle obstacles
  - camera-estimated obstacles or scan rays
  - robot/camera forward direction if practical
- Depth-camera mode completes multiple episodes in simulation.
- Camera mount position and pitch are fixed after testing.
- A final real-install parameter record is produced.

Important measured quantities:

- obstacle body-frame x/y error
- clearance error
- radius estimate error
- front clearance error
- policy action difference between oracle obs and camera-derived obs
- episode outcome in depth-camera mode

## Adapter Debug Pipeline

Keep the adapter inspectable in layers:

1. depth image
2. polar scan
3. obstacle estimates
4. 24D navigation observation
5. navigation action
6. low-level command

The debug mode should make it easy to print or save data at each layer. If the
robot fails in sim, this split helps identify whether the issue is perception,
normalization, policy behavior, or low-level execution.

## Known Risk: Observation Radius Mismatch

The navigation checkpoint metadata says its training environment used:

`observation_radius = 30.0`

The current Isaac config uses:

`observation_radius = 6.0`

This affects the normalized obstacle slots consumed by the policy. Before
calling the D435 adapter sim2real-ready, this mismatch must be understood. The
cleanup should avoid burying this issue behind more script duplication.

## Suggested Implementation Order

1. Update `hierarchical_nav_cfg.py` with a `navigation.depth_camera` config.
2. Extract duplicated script helpers into a small shared module, for example:
   `legged_gym/scripts/hierarchical_nav_demo_utils.py`
3. Convert `play_hierarchical_nav.py` into the first-class entry point with a
   mode selector.
4. Move old scripts into `legged_gym/scripts/legacy/` after the unified entry
   point preserves their behavior.
5. Add pure Python tests or debug utilities for `nav_depth_scan.py` and
   `nav_depth_observation.py` that do not require Isaac Gym.
6. On the Isaac-capable machine, validate in this order:
   - `oracle`
   - `terrain`
   - `depth-scan-debug`
   - `depth-camera`

## Out Of Scope For First Pass

- Retraining the navigation policy.
- Changing the navigation policy input away from 24D.
- Camera mount sweep automation.
- Isaac Gym validation on the current low-GPU machine.
- Large rewrites of `hierarchical_nav.py`.

## Open Questions

- Should the final first-class entry point reuse the exact name
  `play_hierarchical_nav.py`, or should a new `play_hierarchical_nav_demo.py`
  be created first and promoted after validation?
- Should `collision-probe` remain a first-class mode, or move to legacy?
- What exact error threshold is acceptable for camera-derived obstacle estimates
  versus oracle obstacle geometry?
- Should debug data be saved as CSV/NPZ, printed only, or overlaid only in
  video?
