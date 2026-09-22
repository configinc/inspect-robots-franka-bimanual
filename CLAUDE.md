# inspect-robots-franka: agent guide

Inspect Robots adapters for real Franka FR3 and Panda arms: one arm driven by
Physical Intelligence OpenPI DROID policy servers, or a left and right pair
driven by LLM agent policies. The framework lives in
[inspect-robots](https://github.com/robocurve/inspect-robots).

## The one big idea

Inspect Robots swaps a policy and an embodiment. This package ships both:

- `openpi` converts pi05-DROID velocity chunks into absolute joint targets.
- `franka` commands the arm through franky and reads two cameras.
- `franka_bimanual` commands a left and a right arm through two franky drivers
  and reads three cameras. No policy in this package drives it; the LLM agent
  policy from `inspect-robots-agent` binds to its declared spaces.
- `franka_bimanual_robotenv` reuses the same contract and safety behavior over
  franka-controller's left and right RobotEnv gRPC services.

The single-arm pair declares the same 8-D `joint_pos` contract: seven radians
plus one normalized gripper slot, where 0 is closed and 1 is open. The bimanual
embodiment declares that contract twice, left half first, as 16-D. The
`*_bimanual` modules add to the single-arm modules and never change their
contract; `BimanualFrankaConfig.arm_config(side)` slices one side into a
`FrankaConfig` so the single-arm driver factory and validation serve both arms.

## Layout

- `src/inspect_robots_franka/`: package modules and local module map.
- `tests/`: fully injected hardware-free tests.
- `plans/0001-franka-openpi-design.md`: accepted binding design.
- `plans/0002-bimanual-agent-design.md`: accepted two-arm design.

## Working here

- Set `UV_CACHE_DIR=$PWD/.uv-cache` for every uv command in this workspace.
- Install with `uv venv && uv pip install -e ".[dev]"`.
- Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`,
  and `uv run pytest --cov` before handing off.
- Keep strict mypy and 100% statement and branch coverage.
- Keep optional hardware and OpenPI imports lazy so the package imports with
  only Inspect Robots and NumPy.

## Safety invariants

- `FrankaEmbodiment.step()` and `BimanualFrankaEmbodiment.step()` always clamp
  to configured limits without relying on an approver.
- The bimanual embodiment homes and parks one arm at a time, blocking on each,
  and disconnects every connected arm at `close()` even when a park or another
  disconnect fails. It performs no inter-arm collision check.
- The policy integrates DROID velocities before emitting actions. The public
  control mode remains absolute `joint_pos`.
- DROID polarity conversion stays in the policy. The embodiment only sees
  open-positive normalized gripper units.
- Construction performs no hardware, network, camera, or stdin work.
- Success reaches scoring only as `termination_reason="success"`.
- The embodiment declares `SELF_PACED` and sleeps inside `step()`.

## CI and releases

- CI installs from `uv.lock`. Run `uv lock` after dependency changes.
- `ci-ok` must need every blocking job.
- The OpenPI client is installed only from the Physical Intelligence git URL.
- Versions come from git tags through hatch-vcs. Do not add a static project
  version to `pyproject.toml`.

## Writing style

- Do not use em dashes in prose. Use periods, commas, colons, or parentheses.
- Use bold only for definition-list leads and critical safety imperatives.
- Do not use decorative emoji, slogans, chiasmus, or "not just X, but Y".
- Headers use colons, never em dashes or italics.
