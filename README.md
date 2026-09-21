<div align="center">

# inspect-robots-franka

Run [Inspect Robots](https://github.com/robocurve/inspect-robots) evals on real
[Franka FR3](https://franka.de/) and Panda arms, one arm with
[OpenPI](https://github.com/Physical-Intelligence/openpi) DROID policies or a
left and right pair with LLM agent policies.

![Status: alpha](https://img.shields.io/badge/status-alpha-blue)
[![CI](https://github.com/robocurve/inspect-robots-franka/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/inspect-robots-franka/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/inspect-robots-franka)](https://pypi.org/project/inspect-robots-franka/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/robocurve/inspect-robots-franka/actions/workflows/ci.yml)
[![Built on Inspect Robots](https://img.shields.io/badge/built%20on-Inspect%20Robots-indigo)](https://github.com/robocurve/inspect-robots)

</div>

> [!NOTE]
> This project is in early development. Pin a version before depending on its API.

Inspect Robots has two swappable inputs: a `Policy` and an `Embodiment`. This
package provides both sides of a Franka and OpenPI stack, plus a two-arm body:

- **`openpi` policy:** a websocket client for Physical Intelligence OpenPI
  servers, with pi05-DROID velocity integration and gripper conversion.
- **`franka` embodiment:** a lazy franky driver for one FR3 or Panda arm, its
  Franka Hand, and exterior and wrist cameras.
- **`franka_bimanual` embodiment:** two of those drivers for a left and right
  pair, three cameras, and one 16-D contract, built for LLM agent policies.
- **`franka_bimanual_robotenv` embodiment:** the same rig through
  franka-controller, exposed as bounded position-only Cartesian deltas.
  See [Two arms with LLM agent policies](#two-arms-with-llm-agent-policies).

The single-arm pair declares the same 8-D absolute `joint_pos` contract: seven
arm joints in radians followed by a normalized gripper, where 0 is closed and 1
is open. The compatibility check passes with zero errors and zero warnings. For
other robot adapters, see
[inspect-robots-yam](https://github.com/robocurve/inspect-robots-yam) and
[inspect-robots-so101](https://github.com/robocurve/inspect-robots-so101).

## Install:

### Robot machine:

Create an environment and install the package, camera reader, and franky extra:

```bash
uv venv && source .venv/bin/activate
uv pip install "inspect-robots-franka[franka]"
uv pip install "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"
```

`openpi-client` is intentionally installed from the Physical Intelligence git
repository. The unrelated `openpi-client` project on PyPI is not used.

Franky wheels bundle a particular libfranka version. The robot firmware decides
which wheel is compatible. Check the
[franky installation table](https://timschneider42.github.io/franky/) before
connecting, and replace the version selected by the extra when the firmware
requires a different wheel. Enable FCI in the Franka web interface.

Real-time control also needs the Franka host setup: a direct wired connection,
the recommended PREEMPT_RT kernel, correct network settings, and a workstation
that satisfies the libfranka real-time checks. Test the setup with libfranka
examples before running a learned policy.

### GPU machine:

Install OpenPI from its source repository with submodules:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run scripts/serve_policy.py --env=DROID
```

The default DROID server loads `pi05_droid` from
`gs://openpi-assets/checkpoints/pi05_droid` and listens on port 8000. A specific
checkpoint can be served explicitly:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

## Preflight:

Preflight constructs metadata only. It does not connect the robot, cameras, or
policy server.

```bash
inspect-robots-franka-preflight
inspect-robots-franka-preflight --task cubepick-reach
inspect-robots-franka-preflight --dry-run
```

A green report verifies action dimension, control mode, rotation representation,
gripper kind, frame, cameras, state keys, and optional scene realizability. It
cannot infer whether a checkpoint emits velocity or position actions.

`--embodiment franka_bimanual` checks the two-arm body instead, and `--policy
NAME` checks any registered policy: the name resolves through the Inspect Robots
registry with any `-P key=value` arguments forwarded to its constructor and,
when the policy exposes `bind`, is bound to the embodiment's declared spaces
first. Constructing `agent` needs `-P model=provider/model` and that model's API
key in the environment.

## Run on hardware:

The setup wizard interviews the two declared V4L2 camera slots:

```bash
inspect-robots setup
```

The Franka hostname is a network address rather than a discoverable framework
device slot. Add it with the other defaults in
`~/.config/inspect-robots/config.ini`:

```ini
[defaults]
policy = openpi
embodiment = franka
scorer = success_at_end
max_steps = 450
store_frames = true

[policy.args]
host = 192.168.10.20
port = 8000
actions_are_velocity = true
action_horizon = 15
replan_interval = 8

[embodiment.args]
hostname = 172.16.0.2
exterior_cam_device = /dev/v4l/by-id/YOUR-EXTERIOR-CAMERA
wrist_cam_device = /dev/v4l/by-id/YOUR-WRIST-CAMERA
```

Stable `/dev/v4l/by-id/...` or custom udev paths are safer than `/dev/videoN`,
which can change after a replug. With no injected Python camera reader, both
device fields are required. Setting exactly one raises `ConfigError` at
`reset()` before franky connects. An injected reader wins and ignores both
device fields.

Run a task or a direct instruction:

```bash
inspect-robots run --task cubepick-reach --policy openpi --embodiment franka
inspect-robots "pick up the red block" --policy openpi --embodiment franka
```

The attended flow asks the operator to stand clear before homing, then asks for
scene readiness. Press Enter during execution to end the episode and answer y/N
to score it. `FrankaConfig(unattended=True)` skips all prompts and operator
polling. Unattended episodes run until the framework horizon unless another
component ends them.

The upstream websocket client has no inference timeout. A broken or unreachable
server can block `infer()`. Run the server and robot supervisor so a network
stall cannot leave an unsafe scene unattended.

## Two arms with LLM agent policies:

The same package registers `franka_bimanual` for direct franky connections and
`franka_bimanual_robotenv` for the two RobotEnv gRPC services shipped by
configinc/franka-controller. Both represent a left and a right FR3 (or Panda)
pair with three cameras (`exterior_cam`, `left_wrist_cam`, `right_wrist_cam`).
The direct embodiment keeps the 16-D absolute `joint_pos` contract. The
RobotEnv embodiment declares an 8-D `eef_delta_pos` contract: world-frame
`dx`, `dy`, `dz`, and open-positive gripper delta for each arm. Each control
step is limited to 2 cm of translation and 0.2 normalized gripper travel;
end-effector rotation stays fixed. Homing and parking still use the validated
RCI joint reset poses.

No shipped VLA drives it. The released `pi05_droid` checkpoint is single-arm, so
`--policy openpi` against `franka_bimanual` fails compatibility on purpose. The
intended brain is the
[inspect-robots-agent](https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent)
plugin: a frontier LLM builds its whole tool surface from the embodiment's
declared spaces at bind time, so the control contract, per-dimension labels,
and the operating notes in `EmbodimentInfo.docs` are exactly what the model
sees. Swapping models changes only `-P model=`; the robot half never changes.

### Install:

```bash
uv pip install "inspect-robots-franka[franka]" inspect-robots-agent
```

Each arm needs its own FCI connection. libfranka real-time control expects a
direct wired link per robot, so the workstation needs two network interfaces on
two subnets (for example `172.16.0.2` and `172.16.1.2`), FCI enabled on both
robots, and the PREEMPT_RT kernel. Two 1 kHz control loops then share one host.
Validate both arms with the libfranka examples before any learned motion.

For a franka-controller workstation, install the gRPC extra instead and expose
that checkout's generated protocol modules:

```bash
uv pip install "inspect-robots-franka[robotenv]" inspect-robots-agent
export PYTHONPATH=/home/ubuntu/franka-controller:$PYTHONPATH
```

The controller UI remains on port 9000. Complete its six setup steps first.
The Inspect Robots embodiment connects to the left and right RobotEnv services
on ports 50061 and 50063. Ports 50051 and 50053 are the underlying Polymetis
services and are not Inspect Robots endpoints.

### API keys:

The agent policy reads provider keys from the environment. Put them in a `.env`
in the working directory (the CLI loads it) and pick the model with
`-P model=provider/model`:

| Prefix | Key |
|---|---|
| `openai/*` | `OPENAI_API_KEY` |
| `anthropic/*` | `ANTHROPIC_API_KEY` |
| `google/*` | `GEMINI_API_KEY` |
| any model | `OPENROUTER_API_KEY` |

See the agent plugin README for the full provider table and the `-P wire=`
options. This package never reads these keys itself.

### Configure:

```bash
inspect-robots setup
```

The wizard interviews the three declared V4L2 camera slots as one all-or-none
group. Add the two hostnames and the policy defaults by hand in
`~/.config/inspect-robots/config.ini`:

```ini
[defaults]
policy = agent
embodiment = franka_bimanual
scorer = success_at_end
max_steps = 3000
store_frames = true

[policy.args]
model = openai/gpt-6-astra
effort = low
images = on_demand

[embodiment.args]
left_hostname = 172.16.0.2
right_hostname = 172.16.1.2
exterior_cam_device = /dev/v4l/by-id/YOUR-EXTERIOR-CAMERA
left_wrist_cam_device = /dev/v4l/by-id/YOUR-LEFT-WRIST-CAMERA
right_wrist_cam_device = /dev/v4l/by-id/YOUR-RIGHT-WRIST-CAMERA
docs_extra = The arms face each other across a 0.9 m table. The exterior camera looks in from the left arm's side.
```

With franka-controller on the same workstation, select the RobotEnv embodiment
and use gRPC endpoints instead of FCI addresses:

```ini
[defaults]
embodiment = franka_bimanual_robotenv

[embodiment.args]
left_hostname = localhost:50061
right_hostname = localhost:50063
```

RobotEnv uses a closed-positive gripper value while this package exposes an
open-positive value. The adapter converts that polarity in both directions.
This registered profile requires the live services to report `y_frame_v1` with
Robotiq grippers and refuses to move otherwise. Its default `home_pose` and
`rest_pose` are the deployed RCI Y-frame reset joints,
with an open gripper appended to each arm. Override either pose explicitly only
after validating a different Y-frame calibration.

`max_steps` sits far above the single-arm 450 because one agent tool call plays
out as many interpolated steps, up to a 10 s cap per call: 450 steps at 15 Hz is
about three tool calls. `docs_extra` is appended to the notes the model reads.
State where the arms stand relative to each other and to the exterior camera.

### Check without hardware, then run:

```bash
inspect-robots-franka-preflight --embodiment franka_bimanual \
    --policy agent -P model=openai/gpt-6-astra
inspect-robots doctor --embodiment franka_bimanual

# franka-controller transport
inspect-robots-franka-preflight --embodiment franka_bimanual_robotenv \
    --policy agent -P model=openai/gpt-6-astra
inspect-robots doctor --embodiment franka_bimanual_robotenv
```

Preflight constructs the policy with the same `-P` arguments a run would use,
binds it to the embodiment's declared spaces, and reports compatibility.
`doctor` audits the declarations that the agent policy and the default
guardrails depend on. Neither touches hardware. Then:

```bash
inspect-robots "hand the red block from the left arm to the right arm" \
    --policy agent --embodiment franka_bimanual -P model=openai/gpt-6-astra
inspect-robots "stack both cubes on the plate" \
    --policy agent --embodiment franka_bimanual -P model=anthropic/claude-fable-5-1

# after the port 9000 wizard reports both gRPC services healthy
inspect-robots "hand the red block from the left arm to the right arm" \
    --policy agent --embodiment franka_bimanual_robotenv \
    -P model=openai/gpt-6-astra

# use RobotEnv's 16-D absolute joint-position control instead
inspect-robots "hand the red block from the left arm to the right arm" \
    --policy agent --embodiment franka_bimanual_robotenv \
    -E control_mode=joint_pos -P model=openai/gpt-6-astra
```

To test homing, cameras, logs, and video export without an API key, run the
API-free hold policy. It emits zero Cartesian deltas for RobotEnv (or echoes the
observed 16-D joint state for the direct embodiment) and does not interpret the
instruction:

```bash
inspect-robots run --instruction "record the stationary bimanual rig" \
    --policy bimanual_hold --embodiment franka_bimanual_robotenv --max-steps 75
```

Keep the log renderer running beside `inspect-robots view --serve` to create
camera MP4s and replace each completed live page with a video-enabled static
report automatically:

```bash
python scripts/watch_logs.py logs
```

For the agent policy, set `transcript_echo = true` under `[policy.args]` to
also print each model note and tool call in the rollout terminal. The HTML log
records the transcript regardless of that terminal-only setting; provider APIs
do not expose private chain-of-thought.

### Two-arm behavior:

- **One action, both arms.** The direct embodiment clamps a 16-D joint target.
  RobotEnv clamps an 8-D Cartesian-delta command and sends each arm's XYZ delta
  through RCI's existing IK path while holding orientation fixed.
- **Homing and parking are sequential.** `reset()` homes the left arm, then the
  right, each blocking until it arrives, after one stand-clear prompt covering
  both. `close()` parks in the same order and disconnects every connected arm
  even when a park or another disconnect fails.
- **No inter-arm collision check.** Bounds limit each command, not the accumulated
  workspace. Keep the
  workspaces separated, or add a collision check, before unattended runs. The
  embodiment docs tell the model to keep the hands apart and to move one arm at
  a time when they are close.
- **Per-arm gripper gating.** Each hand keeps its own `gripper_deadband` state.
- **Config slicing.** `BimanualFrankaConfig.arm_config("left")` returns the
  `FrankaConfig` that drives one side, so the single-arm driver factory,
  validation rules, and FR3 defaults serve both arms. Validation errors name the
  arm. Panda owners override `joint_low` and `joint_high` for both halves.

### `BimanualFrankaConfig` fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `left_hostname`, `right_hostname` | `None` | FCI addresses, or `host:port` RobotEnv endpoints, both required at `reset()` |
| `joint_low`, `joint_high`, `home_pose`, `rest_pose` | single-arm defaults, tiled | 16-D, left half first |
| `exterior_cam_device`, `left_wrist_cam_device`, `right_wrist_cam_device` | `None` | Builtin OpenCV devices, all-or-none |
| every other field | as `FrankaConfig` | Shared by both arms |

## Safety:

> [!WARNING]
> Keep an operator at the e-stop for initial runs, after firmware changes, and
> whenever a new checkpoint or camera arrangement is introduced.

- **Hard clamp:** every `step()` clips all eight values to
  `FrankaConfig.joint_low/high` inside the embodiment, independent of any
  `Approver`. The defaults are the FR3 datasheet joint bounds pulled 0.05 rad
  inward on each side. Panda limits differ. Panda owners must override them.
- **Velocity semantics:** `pi05_droid` emits normalized joint velocities. The
  policy clips arm velocities to `[-1, 1]`, cumulatively integrates them from
  the latest observed joints, and maps one unit to 0.2 rad per 15 Hz step.
  Position fine-tunes require `actions_are_velocity=false`. Compatibility cannot
  detect a wrong setting. Use preflight, `--dry-run`, and one slow first jog.
- **Control rate:** DROID checkpoints assume 15 Hz execution. Changing
  `FrankaConfig.control_hz` changes the physical velocity represented by every
  integrated policy step.
- **Gripper cadence:** the Franka Hand does not accept a new tracking command at
  every control tick. The embodiment sends a non-blocking move only after the
  normalized target changes by more than `gripper_deadband`, which defaults to
  0.1.
- **Franky requirements:** validate the firmware-specific wheel, FCI state,
  network, real-time kernel, collision thresholds, brakes, and e-stop before
  enabling learned motion.

The CLI's default `DeltaLimitApprover` derives a per-step limit of 5 percent of
each action range. That makes a full gripper stroke a 20-step ramp. The 0.1
embodiment deadband turns the ramp into a bounded command cadence. The same
derived defaults can clip legitimate arm motion. Joint 4 has about a 2.79 rad
configured range, so its default is about 0.14 rad per step, below the policy's
0.2 rad scale. The CLI accepts only a scalar `--max-action-delta` and has no
per-dimension approver override.

Python callers can set a per-dimension delta vector. A practical starting point
allows the DROID arm scale and one complete gripper stroke:

```python
import numpy as np
from inspect_robots import eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots_franka import FrankaEmbodiment, OpenpiPolicy

embodiment = FrankaEmbodiment(
    hostname="172.16.0.2",
    exterior_cam_device="/dev/v4l/by-id/EXTERIOR",
    wrist_cam_device="/dev/v4l/by-id/WRIST",
)
space = embodiment.info.action_space
approver = ChainApprover(
    ClampApprover(space),
    DeltaLimitApprover(space, max_delta=np.asarray([0.2] * 7 + [1.0])),
)
logs = eval("cubepick-reach", OpenpiPolicy(host="192.168.10.20"),
            embodiment, approver=approver)
```

### Gripper polarity:

The embodiment never sees DROID units. Conversion stays at the policy boundary:

| Location | 0 means | 1 means | Conversion |
|----------|---------|---------|------------|
| Franka package wire and embodiment | closed | open | identity |
| Franka Hand width | 0 m | `gripper_max_width` | `wire * max_width` |
| DROID/OpenPI observation and action | open | closed | `wire = 1 - droid` |

For a first run, command asymmetric values such as 0.25 and 0.75 with the arm
stationary. Confirm the observed width and physical motion before evaluating a
policy.

## Configuration:

### Joint-space units:

| Slot | Label | Unit and meaning |
|------|-------|------------------|
| 0 | `joint1` | absolute radians |
| 1 | `joint2` | absolute radians |
| 2 | `joint3` | absolute radians |
| 3 | `joint4` | absolute radians |
| 4 | `joint5` | absolute radians |
| 5 | `joint6` | absolute radians |
| 6 | `joint7` | absolute radians |
| 7 | `gripper` | normalized, 0 closed and 1 open |

The shipped FR3 revolute bounds are:

```text
datasheet low  = (-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159)
datasheet high = ( 2.7437,  1.7837,  2.9007, -0.1518,  2.8065, 4.5169,  3.0159)
shipped low    = datasheet low  + 0.05 rad per arm slot
shipped high   = datasheet high - 0.05 rad per arm slot
```

### `FrankaConfig` fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `hostname` | `None` | FCI address, required at `reset()` |
| `control_hz` | `15.0` | Self-paced command rate |
| `joint_low`, `joint_high` | inset FR3 limits | Absolute hard-clamp bounds |
| `home_pose` | Franka ready pose | Mandatory reset target, gripper open |
| `rest_pose` | `None` | Optional close-time park target; the gripper slot is ignored (arm-only park) |
| `relative_dynamics_factor` | `0.15` | Franky velocity, acceleration, and jerk scale |
| `gripper_max_width` | `0.08` | Physical width represented by wire value 1 |
| `gripper_speed` | `0.05` | Franka Hand speed in m/s |
| `gripper_deadband` | `0.1` | Minimum normalized change before a new hand command |
| `unattended` | `False` | Skip operator readiness and verdict prompts |
| `exterior_cam_device`, `wrist_cam_device` | `None` | Builtin OpenCV camera devices |
| `cam_height`, `cam_width` | `480`, `640` | Declared and returned image resolution |
| `docs_extra` | empty | Rig-specific notes appended to embodiment docs |

### `OpenpiConfig` fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `host`, `port` | `127.0.0.1`, `8000` | OpenPI websocket server |
| `api_key` | `None` | Optional websocket authentication, excluded from eval logs |
| `actions_are_velocity` | `True` | Integrate DROID arm velocity slots |
| `velocity_action_scale` | `0.2` | Radians per step for one normalized velocity unit |
| `action_horizon` | `15` | pi05-DROID chunk length recorded in `PolicyConfig` |
| `replan_interval` | `8` | Actions consumed before framework re-inference |
| `name` | `openpi` | Policy label in eval logs |
| `resize_px` | `224` | Default transport resize-with-pad size |

`pi0_fast_droid` uses an action horizon of 10. Released pi05-DROID uses 15.
Joint-position fine-tunes commonly use 16 and must also set
`actions_are_velocity=false`. A custom injected `infer_fn` owns its image
resizing; `resize_px` applies only to the default websocket transport.

## Development:

Use the local cache path required by this workspace:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
uv venv
uv pip install -e ".[dev]"
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov
```

The suite uses injected drivers, cameras, inference, clocks, and operator I/O.
It requires no robot, server, camera, or stdin. Coverage is enforced at 100
percent with branch coverage enabled.

Dependency changes require `uv lock`. CI uses `uv sync --locked`; the weekly
canary resolves current allowed versions without the lockfile.

## Citation:

```bibtex
@software{inspect-robots-franka,
  author  = {Robocurve},
  title   = {Inspect Robots Franka: OpenPI adapters for Franka arms},
  year    = {2026},
  url     = {https://github.com/robocurve/inspect-robots-franka},
  license = {MIT}
}
```

See [`CITATION.cff`](CITATION.cff) for citation metadata.

## License:

[MIT](LICENSE)
