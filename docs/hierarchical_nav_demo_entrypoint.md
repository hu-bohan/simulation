# Hierarchical Navigation Demo Entry Point

The canonical hierarchical navigation demo entry point is:

```bash
python legged_gym/scripts/play_hierarchical_nav.py --nav-mode oracle
```

`oracle` is also the default mode, so the legacy baseline command remains
recoverable:

```bash
python legged_gym/scripts/play_hierarchical_nav.py
```

## Modes

- `oracle`: loads the low-level locomotion policy, recovery policy, and
  upper-level navigation policy; uses environment-provided oracle obstacle
  slots in the existing 24D navigation observation; applies navigation actions
  through the existing low-level command bridge; does not use D435
  camera-derived obstacle slots.
- `terrain`: loads the same trained policies, enables the static trimesh
  terrain obstacle layout, and uses the environment-synced oracle obstacle
  slots in the existing 24D navigation observation; does not use D435
  camera-derived obstacle slots.
- `depth-scan-debug`: loads the same trained policies, enables terrain
  obstacles and the simulated D435 depth camera, computes camera-derived scans
  and oracle scans side by side, reports clearance differences, and keeps the
  navigation actor on environment-provided obstacle slots.
- `depth-camera`: loads the same trained policies, enables terrain obstacles
  and the simulated D435 depth camera, translates depth data into the existing
  four obstacle slots, feeds the camera-derived 24D navigation observation to
  the upper-level navigation policy, and uses camera-derived front clearance
  for the safety shield while reporting oracle-vs-camera observation/action
  drift.

Run the terrain baseline with:

```bash
python legged_gym/scripts/play_hierarchical_nav.py --nav-mode terrain
```

Run the depth scan debug comparison with:

```bash
python legged_gym/scripts/play_hierarchical_nav.py --nav-mode depth-scan-debug
```

Run the policy-facing D435 mode with:

```bash
python legged_gym/scripts/play_hierarchical_nav.py --nav-mode depth-camera
```

Future modes from the D435 integration plan should be added to
`play_hierarchical_nav.py` rather than creating another first-class demo script.

Operational validation, legacy script locations, and Isaac-capable-machine
run order are documented in:

`docs/runbooks/hierarchical_nav_d435_runbook.md`
