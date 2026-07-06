# Hierarchical Navigation D435 Runbook

This runbook is the operational guide for the consolidated hierarchical
navigation demo entry point and the D435 depth camera integration work.

## Canonical Entry Point

Use this script for new hierarchical navigation demo work:

```bash
python legged_gym/scripts/play_hierarchical_nav.py --nav-mode oracle
```

Supported modes:

- `oracle`: baseline hierarchical navigation. The upper-level navigation policy
  receives environment-provided oracle obstacle slots. No D435 camera-derived
  obstacle slots are used.
- `terrain`: static trimesh obstacle baseline. Terrain obstacles are synced
  into environment oracle obstacle slots. No D435 camera-derived obstacle slots
  are used.
- `depth-scan-debug`: simulated D435 debug path. The mode computes
  camera-derived scans and oracle scans side by side, prints scan differences,
  and can overlay camera scan points in video. The navigation actor still uses
  environment-provided oracle obstacle slots.
- `depth-camera`: policy-facing D435 path. The mode translates D435 depth data
  into the existing four obstacle slots, feeds the resulting 24D navigation
  observation to the trained upper-level navigation actor, and uses
  camera-derived front clearance for the safety shield. It also reports
  oracle-vs-camera observation and action drift.

Run commands:

```bash
python legged_gym/scripts/play_hierarchical_nav.py --nav-mode oracle
python legged_gym/scripts/play_hierarchical_nav.py --nav-mode terrain
python legged_gym/scripts/play_hierarchical_nav.py --nav-mode depth-scan-debug
python legged_gym/scripts/play_hierarchical_nav.py --nav-mode depth-camera
```

The default mode is `oracle`, so this remains valid:

```bash
python legged_gym/scripts/play_hierarchical_nav.py
```

## Legacy Scripts

Older experiments are preserved here:

`legged_gym/scripts/legacy/`

Archived scripts:

- `play_hierarchical_nav_terrain.py`
- `play_hierarchical_nav_depth_scan_debug.py`
- `play_hierarchical_nav_depth_camera.py`
- `play_hierarchical_nav_terrain_collision_probe.py`
- `play_navigation_test.py`
- `play_wjk_pre_demo.py`
- `play_wjk_pre_demo_old.py`

Consult legacy scripts only when recovering historical behavior or comparing
against an old experiment. New behavior should be added to
`legged_gym/scripts/play_hierarchical_nav.py`.

## Collision Probe

The fixed-command collision probe remains a legacy/debug path:

```bash
python legged_gym/scripts/legacy/play_hierarchical_nav_terrain_collision_probe.py
```

Use it only to isolate physical collision or reset behavior from navigation
policy behavior. It is not a normal policy demo mode.

## Local Validation

This machine is not expected to run Isaac Gym. Local validation should focus on
pure Python checks and static checks:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/hubohan/prp/easy_simulation/venv/bin/python -m unittest \
  tests.test_nav_depth_adapter_contract \
  tests.test_nav_depth_scan_debug
```

The adapter contract includes a D435-to-network-input check: a synthetic D435
depth image is translated into a 24D navigation observation and passed through
the trained upper-level navigation actor at `logs/nav/td3_ship_best_actor.pt`.

Static syntax check:

```bash
/home/hubohan/prp/easy_simulation/venv/bin/python -c "from pathlib import Path
paths = [
    'legged_gym/scripts/play_hierarchical_nav.py',
    'legged_gym/scripts/hierarchical_nav_demo_utils.py',
    'legged_gym/utils/nav_depth_scan.py',
    'legged_gym/utils/nav_depth_observation.py',
    'legged_gym/utils/nav_depth_debug.py',
]
for path in paths:
    compile(Path(path).read_text(), path, 'exec')
print('syntax ok')"
```

## Isaac-Capable Machine Validation

Run simulation validation in this order:

1. `oracle`
2. `terrain`
3. `depth-scan-debug`
4. `depth-camera`

The order matters. It separates two-policy bridge problems, terrain obstacle
problems, D435 perception problems, and policy-facing camera input problems.

For `depth-scan-debug`, inspect:

- camera scan front/left/right minima
- oracle scan front/left/right minima
- scan deltas printed by the mode
- video overlay path, goal, oracle obstacles, and camera scan points

For `depth-camera`, inspect:

- camera-derived obstacle count and nearest obstacle report
- camera-derived front clearance
- oracle-vs-camera 24D observation slot drift
- oracle-vs-camera action drift
- command output and reset reasons
- whether multiple episodes complete without obvious perception-induced failure

## Planning Links

- Local PRD: `.scratch/d435-depth-camera-navigation/PRD.md`
- Canonical PRD: `docs/prd/d435-depth-camera-navigation.md`
- Companion plan: `docs/hierarchical_nav_depth_camera_plan.md`
- Canonical entry point notes: `docs/hierarchical_nav_demo_entrypoint.md`
