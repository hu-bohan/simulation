# Agent Instructions

These instructions apply to agents working in this repository.

## First Read

Before changing code, read:

1. `.CONTEXT`
2. `docs/agents/domain.md` if it exists

Treat `.CONTEXT` as stable global context. Treat files under `docs/` and
`.scratch/` as task-specific planning and work-tracking context.

## When The User Says "Complete Issue N"

If the user asks to complete an issue, or gives a path under `.scratch/`, do
this before editing code:

1. Read the referenced issue file.
2. Read the parent PRD referenced by that issue.
3. Read any companion planning document referenced by the PRD.
4. Check the issue's `Blocked by` section.
5. Work only inside the requested issue's scope.

For the current local markdown issue tracker, issues live under:

`.scratch/<feature-slug>/issues/`

PRDs live under:

- `.scratch/<feature-slug>/PRD.md`
- `docs/prd/`

## Scope Discipline

- Do not combine multiple issues unless the user explicitly asks.
- Do not change trained policy interfaces unless the user explicitly asks.
- Do not refactor the low-level locomotion/recovery bridge unless the active
  issue requires it.
- Preserve old demo/test behavior until the relevant issue says it may be
  archived or replaced.
- If existing files have unrelated user changes, do not revert them.

## Local Machine Constraint

This machine is not expected to run Isaac Gym successfully. Missing `isaacgym`
locally is not a project failure.

Use this machine for:

- code edits
- static checks
- pure Python checks
- Torch checkpoint inspection
- documentation and issue work

Isaac Gym simulation validation must happen on another machine.

Use the local venv directly when needed:

`/home/hubohan/prp/easy_simulation/venv/bin/python`

Known local limitations:

- `torch` is available in the venv.
- `isaacgym` is not available in the venv.
- `cv2` is not available in the venv.

## Reporting Completion

When finishing an issue, report:

- files changed
- what was verified locally
- what could not be verified locally because it requires Isaac Gym
- any follow-up needed on the Isaac-capable machine

If the issue file has checkboxes, update them only when the acceptance criteria
are genuinely satisfied.
