# PRD: D435 Depth Camera Navigation Integration

Status: ready-for-agent
Labels: ready-for-agent

## Problem Statement

The project already has trained low-level locomotion/recovery policies and a trained upper-level navigation policy, but the D435 depth camera integration is not organized or validated enough for sim2real transfer.

The current depth-camera work is spread across several play/test scripts with overlapping responsibilities. Some scripts are baselines, some are terrain demos, some compare camera scans against oracle geometry, and one attempts to feed camera-derived obstacle slots into the navigation actor. Because the behavior is scattered, it is hard to know which script is canonical, where to tune camera mounting parameters, and how to determine whether failures come from perception, normalization, policy behavior, or low-level execution.

The user needs a cleaner, documented path from simulated D435 depth data to the upper-level policy's existing 24D navigation observation, while preserving old scripts for reference and avoiding unnecessary changes to the already-working two-policy bridge.

## Solution

Create a single first-class hierarchical navigation entry point that can run the core demo modes, while moving older scripts into a legacy folder for backup. Keep the upper-level policy input unchanged at 24 dimensions, and make D435 depth data produce the existing four obstacle slots expected by the policy.

Move D435 stream parameters and simulated camera mount parameters into the hierarchical navigation config so the camera pose can be tuned in simulation before physical installation. Keep the adapter inspectable in layers: depth image, polar scan, obstacle estimates, 24D navigation observation, navigation action, and low-level command.

Add local documentation and pure Python validation seams so the D435 adapter can be improved on the current machine without requiring Isaac Gym. Isaac Gym validation will happen on another computer with sufficient GPU support.

## User Stories

1. As the project owner, I want one canonical hierarchical navigation demo entry point, so that I know where new navigation demo behavior should be added.
2. As the project owner, I want old test scripts preserved in a legacy folder, so that prior working experiments remain recoverable.
3. As the project owner, I want the most feature-complete script to be the first-class entry point, so that future agents do not chase stale scripts.
4. As a future agent, I want each demo mode to have a clear name, so that I can run the correct behavior without reading every script first.
5. As a future agent, I want an oracle baseline mode, so that I can check the two-policy bridge without depth-camera perception.
6. As a future agent, I want a terrain obstacle mode, so that I can test static trimesh obstacles separately from camera perception.
7. As a future agent, I want a depth scan debug mode, so that I can compare camera-derived scans against oracle obstacle geometry.
8. As a future agent, I want a depth-camera policy mode, so that I can feed camera-derived obstacle slots into the upper-level navigation actor.
9. As a future agent, I want a collision probe mode available if useful, so that I can isolate physical collision behavior from policy behavior.
10. As the project owner, I want D435 parameters in config, so that I can tune camera setup without editing multiple scripts.
11. As the project owner, I want camera mount position in config, so that simulation can guide the real mounting location.
12. As the project owner, I want camera pitch in config, so that I can tune how much floor and obstacle surface the camera sees.
13. As the project owner, I want camera yaw and roll represented even if initially zero, so that future mounting changes do not require interface changes.
14. As the project owner, I want D435 resolution and FOV in config, so that simulated camera data matches the intended real stream profile.
15. As the project owner, I want depth range in config, so that real D435 valid depth behavior can be reflected in simulation.
16. As the project owner, I want scan extraction settings in config, so that ray count, bands, percentiles, and filters can be tuned reproducibly.
17. As the project owner, I want the upper-level policy to keep its 24D input, so that the trained navigation policy remains usable.
18. As the project owner, I want depth data converted into four obstacle slots, so that the D435 adapter matches the trained actor interface.
19. As a future agent, I want the adapter contract documented, so that depth units, camera frame, body frame, normalization, and slot ordering are unambiguous.
20. As a future agent, I want pure Python checks for the depth adapter, so that I can validate geometry without Isaac Gym on the current machine.
21. As a future agent, I want synthetic scan tests, so that known obstacle layouts produce expected obstacle slots.
22. As a future agent, I want camera-derived obstacle estimates compared with oracle obstacles, so that perception errors are measurable.
23. As a future agent, I want camera-derived 24D observations compared with oracle 24D observations, so that policy input drift is visible.
24. As a future agent, I want policy actions compared between oracle and camera-derived observations, so that perception impact on behavior is visible.
25. As the project owner, I want video overlay visualization, so that I can inspect path, goal, oracle obstacles, camera estimates, and robot motion together.
26. As the project owner, I want fixed obstacle validation scenes, so that camera adapter changes are tested against repeatable layouts.
27. As the project owner, I want depth-camera mode to complete multiple episodes in simulation, so that the integration is credible before real-world installation.
28. As the project owner, I want camera mount position and pitch fixed after simulation testing, so that real installation has concrete parameters.
29. As the project owner, I want final real-install parameters recorded, so that the hardware setup can reproduce the simulated assumptions.
30. As a future agent, I want the observation-radius mismatch documented and addressed, so that normalization errors do not masquerade as perception failures.
31. As a future agent, I want low-level policy code left alone unless a concrete bug appears, so that working behavior is not disturbed.
32. As a future agent, I want Isaac Gym validation kept separate from local pure Python work, so that the current low-GPU machine remains useful.
33. As the project owner, I want old script behavior preserved during consolidation, so that cleanup does not erase working demos.
34. As a future agent, I want implementation issues created from this PRD later, so that the work can be split into safe, independently-grabbable slices.

## Implementation Decisions

- The upper-level navigation policy input remains the current 24D vector.
- D435 depth-camera integration converts depth input into the existing four obstacle slots.
- The two-policy bridge is treated as already working and should not be refactored as part of this PRD.
- The final first-class entry point should support oracle baseline, terrain baseline, depth scan debug, depth camera navigation, and optionally collision probe modes.
- Old scripts should be retained in a separate legacy folder rather than deleted.
- D435 stream parameters and mount parameters should move out of play-script constants and into the hierarchical navigation config.
- The depth camera config should include mount body, local position, roll, pitch, yaw, resolution, FOV, depth range, scan parameters, ground filter settings, and obstacle cluster settings.
- The depth adapter should remain layered: depth image to polar scan, polar scan to obstacle estimates, obstacle estimates to 24D observation, observation to navigation action.
- The project should prefer shared helper code for policy loading, environment setup, camera creation, video output, status formatting, and low-level action composition.
- Camera mount sweep automation is intentionally deferred.
- Isaac Gym execution is not expected on the current machine.
- Local markdown is used as the issue tracker for this PRD.

## Testing Decisions

- Test external adapter behavior rather than private implementation details.
- The highest useful test seam is the depth adapter contract: known depth or scan input should produce expected obstacle estimates and expected 24D navigation observation slots.
- Pure Python tests should cover the scanner and translator utilities because they do not require Isaac Gym.
- Synthetic oracle-style scan inputs should be used to validate obstacle slot ordering, body-frame sign conventions, clearance normalization, radius handling, and missing-obstacle slots.
- Debug simulation validation should compare camera-derived scan and obstacle slots against oracle obstacle geometry in fixed known layouts.
- Video overlay validation should show enough scene context to visually compare oracle obstacles and camera-estimated obstacles.
- Integration validation on the Isaac-capable machine should run in this order: oracle baseline, terrain baseline, depth scan debug, then depth camera navigation.
- A depth-camera integration run should be considered credible only after it completes multiple episodes and the camera-derived observation remains close enough to oracle observation in fixed layouts.

## Out of Scope

- Retraining the low-level locomotion policy.
- Retraining the recovery policy.
- Retraining or changing the upper-level navigation policy.
- Changing the upper-level policy input away from the existing 24D observation.
- Large rewrites of the hierarchical navigation environment.
- Isaac Gym validation on the current low-GPU machine.
- Camera mount sweep automation in the first implementation pass.
- Real hardware installation before simulation identifies a suitable mount pose.

## Further Notes

The navigation checkpoint metadata reports `observation_radius = 30.0`, while the current Isaac hierarchical navigation config uses `observation_radius = 6.0`. This affects obstacle-slot normalization and must be explicitly handled before declaring the D435 adapter sim2real-ready.

The local venv for pure Python checks is `/home/hubohan/prp/easy_simulation/venv`. It has CPU Torch but does not have Isaac Gym or OpenCV. Missing Isaac Gym in that venv is expected on this machine.

The companion planning document is `docs/hierarchical_nav_depth_camera_plan.md`.
