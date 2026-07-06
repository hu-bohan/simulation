# Domain Docs

This repo currently uses a single-context documentation layout.

## Before exploring, read these

- `.CONTEXT` at the repo root
- `docs/hierarchical_nav_depth_camera_plan.md`
- Relevant files under `docs/prd/`
- Relevant ADRs under `docs/adr/` if that directory exists later

## Use the project's vocabulary

Use these terms consistently:

- hierarchical navigation
- low-level locomotion policy
- recovery policy
- upper-level navigation policy
- D435 depth camera
- camera-derived obstacle slots
- oracle obstacle slots
- 24D navigation observation
- sim2real

## Flag decision conflicts

If a future proposal contradicts `.CONTEXT`, this domain file, or an ADR, call it out explicitly instead of silently changing direction.
