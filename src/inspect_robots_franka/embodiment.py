"""Real Franka embodiment with lazy hardware, camera, and operator seams.

Every action is hard-clamped inside ``step()`` before reaching franky. The
embodiment owns 15 Hz pacing, uses normalized open-positive gripper units, and
reports operator success only through ``termination_reason="success"``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
from inspect_robots.conformance import DeviceSlot
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_franka import packing
from inspect_robots_franka._franky import FRANKY_INSTALL_COMMAND, _load_franky
from inspect_robots_franka.config import FrankaConfig, action_box, observation_space
from inspect_robots_franka.operator import OperatorIO, default_poll_end

Vec = npt.NDArray[np.float64]
ImageMap = Mapping[str, npt.NDArray[np.uint8]]

_DOCS = """A single 7-DoF Franka arm with a parallel-jaw hand. Actions and state use
absolute joint positions in the robot base frame. Revolute joints are radians;
the hand is normalized with 0 closed and 1 open.
- joint1: base rotation about the vertical axis.
- joint2: first shoulder pitch joint.
- joint3: upper-arm roll joint.
- joint4: elbow pitch joint.
- joint5: forearm roll joint.
- joint6: wrist pitch joint.
- joint7: wrist roll joint.
- gripper: parallel-jaw opening, with 0 fully closed and 1 fully open.
Use small changes, keep people clear of the workspace, and re-check camera and
proprioceptive observations after every deliberate motion."""


@runtime_checkable
class TaskEnvelopeLike(Protocol):
    """Read-only rollout metadata accepted by the optional task-binding hook."""

    @property
    def max_steps(self) -> int:
        """Return the framework-enforced rollout horizon."""
        ...


@runtime_checkable
class Driver(Protocol):
    """Minimal Franka arm and hand surface needed by the embodiment."""

    def read_joints(self) -> npt.NDArray[np.floating[Any]]:
        """Read seven revolute joint positions in radians."""
        ...

    def read_gripper_width(self) -> float:
        """Read the physical finger width in metres."""
        ...

    def move_joints(self, target: npt.NDArray[np.floating[Any]]) -> None:
        """Asynchronously preempt the current arm target."""
        ...

    def move_joints_sync(self, target: npt.NDArray[np.floating[Any]]) -> None:
        """Block until a homing or parking arm target completes."""
        ...

    def move_gripper(self, width: float) -> None:
        """Asynchronously command a physical finger width in metres."""
        ...

    def disconnect(self) -> None:
        """Release arm and hand resources."""
        ...


DriverFactory = Callable[[FrankaConfig], Driver]
CameraReader = Callable[[], ImageMap]


def _default_driver_factory(cfg: FrankaConfig) -> Driver:  # pragma: no cover - real hardware
    franky = _load_franky()
    hostname = cast(str, cfg.hostname)
    robot = franky.Robot(hostname)
    hand = franky.Gripper(hostname)
    robot.relative_dynamics_factor = cfg.relative_dynamics_factor

    class _RealDriver:
        def read_joints(self) -> npt.NDArray[np.floating[Any]]:
            return np.asarray(robot.current_joint_state.position)

        def read_gripper_width(self) -> float:
            return float(hand.width)

        def move_joints(self, target: npt.NDArray[np.floating[Any]]) -> None:
            robot.move(franky.JointMotion(np.asarray(target).tolist()), asynchronous=True)

        def move_joints_sync(self, target: npt.NDArray[np.floating[Any]]) -> None:
            robot.move(franky.JointMotion(np.asarray(target).tolist()))

        def move_gripper(self, width: float) -> None:
            hand.move_async(width, cfg.gripper_speed)

        def disconnect(self) -> None:
            hand.stop()
            robot.stop()

    return _RealDriver()


def _opencv_camera_reader(cfg: FrankaConfig) -> CameraReader:
    """Build a lazy OpenCV reader for the two configured V4L2 devices."""
    return opencv_camera_reader(
        {"exterior_cam": cfg.exterior_cam_device, "wrist_cam": cfg.wrist_cam_device},
        width=cfg.cam_width,
        height=cfg.cam_height,
    )


def opencv_camera_reader(
    devices: Mapping[str, str | int | None], *, width: int, height: int
) -> CameraReader:
    """Build a lazy OpenCV reader for named V4L2 devices at one shared resolution.

    Nothing is opened until the first read, so construction stays hardware-free.
    """
    captures: dict[str, Any] = {}

    def reader() -> ImageMap:  # pragma: no cover - real cameras
        import cv2

        if not captures:
            for name, device in devices.items():
                cap = cv2.VideoCapture(cast(Any, device))
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open {name} at {device}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                captures[name] = cap
        frames: dict[str, npt.NDArray[np.uint8]] = {}
        for name, cap in captures.items():
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"frame read failed for {name} ({devices[name]})")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (width, height))
            frames[name] = np.asarray(rgb, dtype=np.uint8)
        return frames

    return reader


class FrankaEmbodiment:
    """Inspect Robots embodiment for FR3 or compatible Panda joint control."""

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
            arg="wrist_cam_device",
            kind="v4l2",
            label="wrist camera",
            group="cameras",
        ),
    )

    def __init__(
        self,
        config: FrankaConfig | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        camera_reader: CameraReader | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else FrankaConfig.from_kwargs(**flat)
        self._driver_factory: DriverFactory = driver_factory or _default_driver_factory
        self._camera_reader = camera_reader
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end: Callable[[], bool] = poll_end or default_poll_end
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._sleep: Callable[[float], None] = sleep_fn or time.sleep
        self._driver: Driver | None = None
        self._instruction: str | None = None
        self._last_gripper_command: float | None = None
        self._t_last = 0.0
        self._bound_max_steps: int | None = None
        self.num_steps = 0

        docs = _DOCS
        docs_extra = self._cfg.docs_extra.strip()
        if docs_extra:
            docs += "\n\n" + docs_extra
        self.info = EmbodimentInfo(
            name="franka",
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
        """Validate camera setup, connect lazily, home, and confirm readiness."""
        reader = self._camera_reader
        devices = (self._cfg.exterior_cam_device, self._cfg.wrist_cam_device)
        if reader is not None and not callable(reader):
            raise ConfigError("camera_reader must be callable when injected")
        if reader is None:
            if sum(device is not None for device in devices) != 2:
                raise ConfigError(
                    "franka cameras require both exterior_cam_device and "
                    "wrist_cam_device when no camera_reader is injected"
                )
            reader = _opencv_camera_reader(self._cfg)
            self._camera_reader = reader
        if self._cfg.hostname is None:
            raise ConfigError(
                "FrankaConfig.hostname is required before reset; set the robot FCI address"
            )
        if self._driver is None:
            self._driver = self._driver_factory(self._cfg)
        if not self._cfg.unattended:
            self._operator.wait_ready(
                "Arm will move to the home pose. Stand clear, then press Enter..."
            )
        home = packing.validate_dim(self._cfg.home_pose)
        self._driver.move_joints_sync(packing.arm_joints(home))
        self._driver.move_gripper(packing.gripper(home) * self._cfg.gripper_max_width)
        self._last_gripper_command = packing.gripper(home)
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
        """Clamp, command, pace, observe, and optionally collect a verdict."""
        driver = self._require_driver()
        self.num_steps += 1
        command = packing.validate_dim(action.data)
        clamped = np.clip(command, self._cfg.low, self._cfg.high)
        driver.move_joints(packing.arm_joints(clamped))
        gripper_target = packing.gripper(clamped)
        if (
            self._last_gripper_command is None
            or abs(gripper_target - self._last_gripper_command) > self._cfg.gripper_deadband
        ):
            driver.move_gripper(gripper_target * self._cfg.gripper_max_width)
            self._last_gripper_command = gripper_target
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
        """Optionally park the arm, then disconnect and clear the handle even on error.

        The park is arm-only: a non-blocking gripper command issued here would
        race the immediate disconnect on real hardware, so the gripper slot of
        ``rest_pose`` is deliberately ignored.
        """
        self._bound_max_steps = None
        driver = self._driver
        if driver is None:
            return
        try:
            if self._cfg.rest_pose is not None:
                rest = packing.validate_dim(self._cfg.rest_pose)
                clamped = np.clip(rest, self._cfg.low, self._cfg.high)
                driver.move_joints_sync(packing.arm_joints(clamped))
        finally:
            try:
                driver.disconnect()
            finally:
                self._driver = None
                self._last_gripper_command = None

    def _require_driver(self) -> Driver:
        """Return the connected driver or reject step-before-reset use."""
        if self._driver is None:
            raise RuntimeError("step() called before reset() (or after close())")
        return self._driver

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
        """Read cameras and canonical open-positive joint state."""
        driver = self._require_driver()
        joints = np.asarray(driver.read_joints(), dtype=np.float64)
        if joints.ndim != 1 or joints.shape[0] != packing.NUM_JOINTS:
            raise ValueError(
                f"driver returned joints of shape {joints.shape}; expected ({packing.NUM_JOINTS},)"
            )
        normalized_width = float(
            np.clip(driver.read_gripper_width() / self._cfg.gripper_max_width, 0.0, 1.0)
        )
        state = np.concatenate((joints, np.asarray([normalized_width])))
        reader = self._camera_reader
        if reader is None:  # pragma: no cover - reset always installs one
            raise RuntimeError("camera reader unavailable before reset")
        images = {name: np.asarray(frame, dtype=np.uint8) for name, frame in reader().items()}
        observed_at = self._clock()
        return Observation(
            images=images,
            state={packing.STATE_KEY: state},
            instruction=instruction,
            image_times=dict.fromkeys(images, observed_at),
            state_time=observed_at,
        )
