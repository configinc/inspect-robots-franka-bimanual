"""Real two-arm Franka embodiment built from two single-arm franky drivers.

One 16-D action commands both arms in the same control tick: the left half goes
to the left driver and the right half to the right driver, each hard-clamped
inside ``step()`` before reaching franky. Homing and parking run one arm at a
time and block until that arm arrives. Pacing, gripper gating, operator prompts,
and success reporting follow the single-arm embodiment.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, ClassVar

import numpy as np
from inspect_robots.conformance import DeviceSlot
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_franka import packing, packing_bimanual
from inspect_robots_franka._franky import FRANKY_INSTALL_COMMAND
from inspect_robots_franka.config_bimanual import (
    BimanualFrankaConfig,
    action_box,
    observation_space,
)
from inspect_robots_franka.embodiment import (
    CameraReader,
    Driver,
    DriverFactory,
    TaskEnvelopeLike,
    _default_driver_factory,
    opencv_camera_reader,
)
from inspect_robots_franka.operator import OperatorIO, default_poll_end
from inspect_robots_franka.packing_bimanual import ARMS

_DOCS = """Two 7-DoF Franka arms, a left arm and a right arm, each with a parallel-jaw
hand. One action commands both arms at once: the first eight slots belong to the
left arm and the last eight to the right arm. Actions and state use absolute joint
positions, each arm in its own base frame. Revolute joints are radians; each hand
is normalized with 0 closed and 1 open.
- left_joint1: left arm base rotation about the vertical axis.
- left_joint2: left arm first shoulder pitch joint.
- left_joint3: left arm upper-arm roll joint.
- left_joint4: left arm elbow pitch joint.
- left_joint5: left arm forearm roll joint.
- left_joint6: left arm wrist pitch joint.
- left_joint7: left arm wrist roll joint.
- left_gripper: left parallel-jaw opening, with 0 fully closed and 1 fully open.
- right_joint1: right arm base rotation about the vertical axis.
- right_joint2: right arm first shoulder pitch joint.
- right_joint3: right arm upper-arm roll joint.
- right_joint4: right arm elbow pitch joint.
- right_joint5: right arm forearm roll joint.
- right_joint6: right arm wrist pitch joint.
- right_joint7: right arm wrist roll joint.
- right_gripper: right parallel-jaw opening, with 0 fully closed and 1 fully open.
To hold one arm still, repeat its current joint values while moving the other.
The arms share a workspace and nothing stops them from colliding with each other:
keep the hands apart unless the task needs a handover, and move one arm at a time
when they are close. Where each arm stands relative to the other and to the
exterior camera is rig-specific; read any rig notes below before planning a
two-arm motion. Use small changes, keep people clear of the workspace, and
re-check camera and proprioceptive observations after every deliberate motion."""


def _disconnect_all(drivers: Iterable[Driver]) -> None:
    """Disconnect every driver, then re-raise the first failure if any."""
    first: Exception | None = None
    for driver in drivers:
        try:
            driver.disconnect()
        except Exception as exc:
            if first is None:
                first = exc
    if first is not None:
        raise first


class BimanualFrankaEmbodiment:
    """Inspect Robots embodiment for a left and right FR3 or Panda pair."""

    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "franky": FRANKY_INSTALL_COMMAND,
        "cv2": "pip install opencv-python-headless",
    }
    DEVICE_SLOTS: ClassVar[tuple[DeviceSlot, ...]] = (
        DeviceSlot(
            arg="exterior_cam_device",
            kind="v4l2",
            label="exterior camera",
            group="cameras",
        ),
        DeviceSlot(
            arg="left_wrist_cam_device",
            kind="v4l2",
            label="left wrist camera",
            group="cameras",
        ),
        DeviceSlot(
            arg="right_wrist_cam_device",
            kind="v4l2",
            label="right wrist camera",
            group="cameras",
        ),
    )

    def __init__(
        self,
        config: BimanualFrankaConfig | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        camera_reader: CameraReader | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else BimanualFrankaConfig.from_kwargs(**flat)
        self._driver_factory: DriverFactory = driver_factory or _default_driver_factory
        self._camera_reader = camera_reader
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end: Callable[[], bool] = poll_end or default_poll_end
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._sleep: Callable[[float], None] = sleep_fn or time.sleep
        self._drivers: dict[str, Driver] = {}
        self._instruction: str | None = None
        self._last_gripper_command: dict[str, float | None] = dict.fromkeys(ARMS)
        self._t_last = 0.0
        self._bound_max_steps: int | None = None
        self.num_steps = 0

        docs = _DOCS
        docs_extra = self._cfg.docs_extra.strip()
        if docs_extra:
            docs += "\n\n" + docs_extra
        self.info = EmbodimentInfo(
            name="franka_bimanual",
            action_space=action_box(self._cfg),
            observation_space=observation_space(self._cfg),
            control_hz=self._cfg.control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED}),
            docs=docs,
        )

    def bind_task(self, envelope: TaskEnvelopeLike) -> None:
        """Store the rollout horizon for operator-facing integrations."""
        self._bound_max_steps = int(envelope.max_steps)

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Validate camera setup, connect both arms lazily, home them, and confirm readiness."""
        reader = self._camera_reader
        devices = self._cfg.camera_devices
        if reader is not None and not callable(reader):
            raise ConfigError("camera_reader must be callable when injected")
        if reader is None:
            if any(device is None for device in devices.values()):
                raise ConfigError(
                    "franka_bimanual cameras require exterior_cam_device, "
                    "left_wrist_cam_device, and right_wrist_cam_device when no "
                    "camera_reader is injected"
                )
            reader = opencv_camera_reader(
                devices, width=self._cfg.cam_width, height=self._cfg.cam_height
            )
            self._camera_reader = reader
        missing = [side for side in ARMS if getattr(self._cfg, f"{side}_hostname") is None]
        if missing:
            names = " and ".join(f"{side}_hostname" for side in missing)
            raise ConfigError(
                f"BimanualFrankaConfig.{names} required before reset; set both FCI addresses"
            )
        for side in ARMS:
            if side not in self._drivers:
                self._drivers[side] = self._driver_factory(self._cfg.arm_config(side))
        if not self._cfg.unattended:
            self._operator.wait_ready(
                "Both arms will move to the home pose. Stand clear, then press Enter..."
            )
        homes = packing_bimanual.split_arms(self._cfg.home_pose)
        for side, home in zip(ARMS, homes, strict=True):
            driver = self._drivers[side]
            driver.move_joints_sync(packing.arm_joints(home))
            driver.move_gripper(packing.gripper(home) * self._cfg.gripper_max_width)
            self._last_gripper_command[side] = packing.gripper(home)
        if not self._cfg.unattended:
            self._operator.wait_ready()
            horizon = self._horizon_seconds()
            limit = f" Max {horizon:.0f}s." if horizon is not None else ""
            self._operator.output_fn(
                "Running: press Enter to end the episode, then y/N to score." + limit
            )
        self._instruction = scene.instruction
        self.num_steps = 0
        self._t_last = self._clock()
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        """Clamp, command both arms, pace, observe, and optionally collect a verdict."""
        drivers = self._require_drivers()
        self.num_steps += 1
        command = packing_bimanual.validate_dim(action.data)
        clamped = np.clip(command, self._cfg.low, self._cfg.high)
        for side, part in zip(ARMS, packing_bimanual.split_arms(clamped), strict=True):
            driver = drivers[side]
            driver.move_joints(packing.arm_joints(part))
            gripper_target = packing.gripper(part)
            last = self._last_gripper_command[side]
            if last is None or abs(gripper_target - last) > self._cfg.gripper_deadband:
                driver.move_gripper(gripper_target * self._cfg.gripper_max_width)
                self._last_gripper_command[side] = gripper_target
        self._pace()
        observation = self._observe(self._instruction)
        if not self._cfg.unattended and self._poll_end():
            success = self._operator.confirm_success()
            return StepResult(
                observation=observation,
                terminated=True,
                termination_reason="success" if success else "failure",
                info={"operator_confirmed": success},
            )
        return StepResult(observation=observation, terminated=False)

    def close(self) -> None:
        """Optionally park each connected arm, then disconnect all of them.

        Parking is arm-only and sequential. Every connected driver is
        disconnected even when parking or another disconnect fails, and the
        handles are cleared so a later ``reset()`` reconnects from scratch.
        """
        self._bound_max_steps = None
        drivers = dict(self._drivers)
        if not drivers:
            return
        try:
            if self._cfg.rest_pose is not None:
                rest = packing_bimanual.validate_dim(self._cfg.rest_pose)
                clamped = np.clip(rest, self._cfg.low, self._cfg.high)
                for side, part in zip(ARMS, packing_bimanual.split_arms(clamped), strict=True):
                    driver = drivers.get(side)
                    if driver is not None:
                        driver.move_joints_sync(packing.arm_joints(part))
        finally:
            try:
                _disconnect_all(drivers.values())
            finally:
                self._drivers.clear()
                self._last_gripper_command = dict.fromkeys(ARMS)

    def _require_drivers(self) -> Mapping[str, Driver]:
        """Return both connected drivers or reject step-before-reset use."""
        if len(self._drivers) != len(ARMS):
            raise RuntimeError("step() called before reset() (or after close())")
        return self._drivers

    def _pace(self) -> None:
        """Sleep for the remainder of the configured control period."""
        elapsed = self._clock() - self._t_last
        self._sleep(max(0.0, 1.0 / self._cfg.control_hz - elapsed))
        self._t_last = self._clock()

    def _horizon_seconds(self) -> float | None:
        """Return the bound rollout horizon in self-paced wall-clock seconds."""
        if self._bound_max_steps is None:
            return None
        return self._bound_max_steps / self._cfg.control_hz

    def _observe(self, instruction: str | None) -> Observation:
        """Read cameras and the packed open-positive joint state of both arms."""
        drivers = self._require_drivers()
        halves: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        for side in ARMS:
            driver = drivers[side]
            joints = np.asarray(driver.read_joints(), dtype=np.float64)
            if joints.ndim != 1 or joints.shape[0] != packing.NUM_JOINTS:
                raise ValueError(
                    f"{side} driver returned joints of shape {joints.shape}; "
                    f"expected ({packing.NUM_JOINTS},)"
                )
            normalized_width = float(
                np.clip(driver.read_gripper_width() / self._cfg.gripper_max_width, 0.0, 1.0)
            )
            halves[side] = np.concatenate((joints, np.asarray([normalized_width])))
        state = packing_bimanual.join_arms(halves["left"], halves["right"])
        reader = self._camera_reader
        if reader is None:  # pragma: no cover - reset always installs one
            raise RuntimeError("camera reader unavailable before reset")
        images = {name: np.asarray(frame, dtype=np.uint8) for name, frame in reader().items()}
        observed_at = self._clock()
        return Observation(
            images=images,
            state={packing_bimanual.STATE_KEY: state},
            instruction=instruction,
            image_times=dict.fromkeys(images, observed_at),
            state_time=observed_at,
        )
