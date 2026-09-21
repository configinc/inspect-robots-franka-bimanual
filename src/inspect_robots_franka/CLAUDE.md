# `inspect_robots_franka` package: module map

The package supplies the `franka` embodiment, the `openpi` policy, the glue that
keeps their 8-D absolute joint-position contract symmetric, and the
`franka_bimanual` embodiment that repeats the contract for a left and right pair.

## Modules

| Module | Responsibility |
|--------|----------------|
| `packing.py` | Pure constants, strict 8-D validation, and arm/gripper accessors. |
| `config.py` | Frozen configs and the shared action and observation space builders. |
| `policy.py` | Lazy OpenPI websocket client, DROID gripper conversion, and velocity integration. |
| `embodiment.py` | Lazy franky driver, hard clamp, gripper gating, cameras, pacing, and success verdicts. |
| `packing_bimanual.py` | 16-D constants, left-first labels, strict validation, and split/join helpers over `packing.py`. |
| `config_bimanual.py` | Frozen two-arm config that slices into per-arm `FrankaConfig`, plus the 16-D space builders. |
| `embodiment_bimanual.py` | Two per-arm drivers behind one embodiment: sequential homing, same-tick commands, per-arm gripper gating. |
| `_franky.py` | Guided loader for the optional firmware-specific franky wheel. |
| `operator.py` | Injectable readiness and scoring prompts. |
| `preflight.py` | Hardware-free compatibility CLI for either embodiment and any registered policy. |
| `__init__.py` | Reviewed public API, fenced by `__all__`. |

## Invariants

- Construction performs no hardware or network I/O.
- `step()` clamps every command independently of framework approvers.
- The policy converts DROID velocities into absolute targets before returning.
- The embodiment only handles normalized open-positive gripper units.
- Only `termination_reason="success"` reports success to a scorer.
- Bimanual modules import the single-arm modules and never change them; the
  single-arm 8-D contract stays the source of truth for each half.
