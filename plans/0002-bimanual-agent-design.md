# 0002: Bimanual Franka embodiment for LLM agent policies

Status: accepted (2026-09-21)
Fork: configinc/inspect-robots-franka-bimanual, on top of upstream `cb46d74`.

## Goal

Register a second embodiment, `franka_bimanual`, so a left and right FR3 (or
Panda) pair can be evaluated with frontier LLM agent policies
(`inspect-robots-agent`, `--policy agent -P model=provider/model`) through the
unchanged Inspect Robots eval loop. Any provider the agent plugin supports works
by swapping `-P model=`; this package reads no API key itself.

## Decisions

- **Add beside, never modify.** The 8-D modules stay the source of truth for one
  arm. `packing_bimanual.py`, `config_bimanual.py`, and
  `embodiment_bimanual.py` build on them. The only single-arm edit is the
  OpenCV camera reader, generalized to a named device map so both embodiments
  share it. Single-arm behavior, tests, and the `franka` entry point are
  untouched.
- **16-D contract, left half first.** `DIM_LABELS = (left_joint1, ...,
  left_gripper, right_joint1, ..., right_gripper)`. Absolute `joint_pos`, one
  `StateSpec` field shaped `(16,)`, finite FR3 bounds tiled per arm. This is
  exactly what `check_embodiment` requires for the agent policy to build its
  tool surface, and what `inspect-robots-yam` does at 14-D.
- **Slice, do not duplicate, the hardware layer.**
  `BimanualFrankaConfig.arm_config(side)` returns a `FrankaConfig` for one side,
  so the single-arm `_default_driver_factory` (franky wheel loading, FCI
  connect, dynamics factor, gripper speed) and every single-arm validation rule
  serve both arms. Validation errors are prefixed with the arm.
- **Same-tick commands, sequential homing.** `step()` splits the clamped 16-D
  command and issues both asynchronous franky targets in one tick, then paces.
  `reset()` and `close()` home and park left then right, blocking on each, after
  one stand-clear prompt. `close()` disconnects every connected arm even if a
  park or an earlier disconnect fails, and clears the handles.
- **Three cameras, one all-or-none group.** `exterior_cam`, `left_wrist_cam`,
  `right_wrist_cam` as V4L2 device slots in group `cameras`, so the setup wizard
  interviews them together. Hostnames stay config-file fields, as upstream.
- **No shipped bimanual policy.** `pi05_droid` is single-arm, so `openpi`
  against `franka_bimanual` is a deliberate `action_dim` error. Preflight gained
  `--embodiment` and `--policy NAME`; a registry policy exposing `bind` is bound
  to the embodiment's spaces before the check, mirroring the agent plugin.

## Out of scope

- Inter-arm collision checking. The hard clamp is a per-joint box; the docs
  handed to the model say so and ask it to keep the hands apart.
- Trajectory synchronization between arms beyond the shared control tick.
- Cartesian or delta control modes, per-arm control rates, and bimanual VLA
  adapters (XPolicyLab-served policies would arrive as separate plugins).
- Renaming the distribution. The fork keeps `inspect-robots-franka` as the
  package and import name; installing it beside upstream in one environment
  collides.

## Verification

- 100 percent statement and branch coverage, strict mypy, ruff, on the locked
  `inspect-robots==0.21.0`.
- `check_embodiment(BimanualFrankaEmbodiment().info)` has zero issues and a
  bound 16-D policy is compatible with zero errors and zero warnings.
- Manually confirmed against `inspect-robots==0.58.0` and
  `inspect-robots-agent==0.26.0`: `LLMAgentPolicy.bind()` adopts the 16-D
  contract and `check_compatibility` passes for OpenAI, Anthropic, and Gemini
  model strings.
