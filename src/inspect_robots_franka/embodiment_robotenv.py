"""Bimanual Franka embodiment backed by franka-controller RobotEnv gRPC."""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, cast

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.types import Action, StepResult

from inspect_robots_franka.config import FrankaConfig
from inspect_robots_franka.config_bimanual import BimanualFrankaConfig
from inspect_robots_franka.embodiment import Driver, DriverFactory
from inspect_robots_franka.embodiment_bimanual import BimanualFrankaEmbodiment
from inspect_robots_franka.packing_bimanual import ARMS

GRPC_INSTALL_COMMAND = 'pip install "inspect-robots-franka[robotenv]"'
PROTO_INSTALL_COMMAND = "export PYTHONPATH=/home/ubuntu/franka-controller:$PYTHONPATH"
Y_FRAME_ROBOTIQ_HOME_POSE: tuple[float, ...] = (
    0.0122,
    -0.0490,
    0.0695,
    -2.4348,
    1.8865,
    2.6217,
    -0.7963,
    1.0,
    -0.1657,
    -0.0753,
    0.0752,
    -2.4377,
    -1.8362,
    2.6705,
    0.9437,
    1.0,
)
ROBOTENV_DIM_LABELS: tuple[str, ...] = tuple(
    f"{side}_{label}" for side in ARMS for label in ("dx", "dy", "dz", "gripper")
)
ROBOTENV_ACTION_LOW = np.asarray((-0.02, -0.02, -0.02, -0.2) * len(ARMS), dtype=np.float64)
ROBOTENV_ACTION_HIGH = -ROBOTENV_ACTION_LOW
ROBOTENV_ACTION_SEMANTICS = ActionSemantics(
    control_mode="eef_delta_pos",
    rotation_repr="none",
    gripper="continuous",
    frame="world",
    dim_labels=ROBOTENV_DIM_LABELS,
)


def robotenv_action_box() -> Box:
    """Return the bounded bimanual Cartesian-delta action contract."""
    return Box(
        shape=(len(ROBOTENV_DIM_LABELS),),
        low=ROBOTENV_ACTION_LOW.copy(),
        high=ROBOTENV_ACTION_HIGH.copy(),
        semantics=ROBOTENV_ACTION_SEMANTICS,
    )


class CartesianDriver(Driver, Protocol):
    """RobotEnv extension used for position-only Cartesian deltas."""

    def move_cartesian_delta(
        self, position_delta: npt.NDArray[np.floating[Any]], gripper_open_delta: float
    ) -> None:
        """Move by XYZ in the world frame while holding end-effector rotation."""


def _load_robotenv() -> tuple[Any, Any, Any]:  # pragma: no cover - live optional imports
    """Load the optional gRPC runtime and franka-controller generated stubs."""
    try:
        grpc = importlib.import_module("grpc")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"RobotEnv gRPC support is optional. Install it with: {GRPC_INSTALL_COMMAND}"
        ) from exc
    try:
        pb2 = importlib.import_module("proto.robotenv_pb2")
        pb2_grpc = importlib.import_module("proto.robotenv_pb2_grpc")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "franka-controller RobotEnv stubs are not importable. Set its checkout on "
            f"PYTHONPATH, for example: {PROTO_INSTALL_COMMAND}"
        ) from exc
    return grpc, pb2, pb2_grpc


class _RobotEnvDriver:
    """Adapt one RobotEnv service to the package's small arm driver protocol."""

    def __init__(
        self,
        cfg: FrankaConfig,
        *,
        modules: tuple[Any, Any, Any] | None = None,
    ) -> None:
        if cfg.hostname is None:
            raise ValueError("RobotEnv endpoint is required")
        grpc, self._pb2, pb2_grpc = modules or _load_robotenv()
        self._cfg = cfg
        self._channel = grpc.insecure_channel(cfg.hostname)
        self._stub = pb2_grpc.RobotEnvStub(self._channel)
        self._joints: npt.NDArray[np.float64] | None = None
        self._gripper_open: float | None = None
        self._gripper_open_command: float | None = None

        robot_cfg = self._stub.GetConfig(self._pb2.GetConfigRequest(), timeout=5.0)
        required = {"cartesian_delta", "joint_position"}
        missing = required.difference(robot_cfg.supported_action_spaces)
        if missing:
            raise RuntimeError(
                f"RobotEnv at {cfg.hostname} does not support {', '.join(sorted(missing))}"
            )
        if (robot_cfg.frame_type, robot_cfg.gripper_type) != ("y_frame_v1", "robotiq"):
            raise RuntimeError(
                f"RobotEnv at {cfg.hostname} is {robot_cfg.frame_type} + "
                f"{robot_cfg.gripper_type}; expected y_frame_v1 + robotiq"
            )
        health = self._stub.HealthCheck(self._pb2.HealthCheckRequest(), timeout=5.0)
        if health.status != "HEALTHY":
            raise RuntimeError(f"RobotEnv at {cfg.hostname} is {health.status}: {health.message}")

    def _remember(self, observation: Any) -> None:
        joints = np.asarray(observation["joint_positions"].float_array.values, dtype=np.float64)
        if joints.shape != (7,):
            raise ValueError(f"RobotEnv returned joints of shape {joints.shape}; expected (7,)")
        gripper = observation["gripper_position"]
        if gripper.HasField("float_value"):
            closed = float(gripper.float_value)
        elif gripper.HasField("float_array") and len(gripper.float_array.values) == 1:
            closed = float(gripper.float_array.values[0])
        else:
            raise ValueError("RobotEnv gripper_position must be one normalized value")
        self._joints = joints
        self._gripper_open = float(np.clip(1.0 - closed, 0.0, 1.0))

    @staticmethod
    def _check(response: Any, operation: str) -> None:
        if response.status != "SUCCESS":
            raise RuntimeError(
                f"RobotEnv {operation} failed: {response.status}: {response.message}"
            )

    def _refresh(self) -> None:
        response = self._stub.Reset(self._pb2.ResetRequest(mode="current"), timeout=5.0)
        self._check(response, "current-state read")
        self._remember(response.observation)
        if self._gripper_open_command is None:
            self._gripper_open_command = self._gripper_open

    def _step(self, action: list[float], action_space: str) -> None:
        response = self._stub.Step(
            self._pb2.StepRequest(
                action=action,
                action_space=action_space,
                gripper_action_space="position",
            ),
            timeout=5.0,
        )
        self._check(response, "step")
        self._remember(response.observation)

    def read_joints(self) -> npt.NDArray[np.float64]:
        if self._joints is None:
            self._refresh()
        joints = self._joints
        assert joints is not None
        return cast(npt.NDArray[np.float64], joints.copy())

    def read_gripper_width(self) -> float:
        if self._gripper_open is None:
            self._refresh()
        assert self._gripper_open is not None
        return self._gripper_open * self._cfg.gripper_max_width

    def move_joints(self, target: npt.NDArray[np.floating[Any]]) -> None:
        if self._gripper_open is None:
            self._refresh()
        assert self._gripper_open is not None
        action = [*np.asarray(target, dtype=np.float64).tolist(), 1.0 - self._gripper_open]
        self._step(action, "joint_position")

    def move_joints_sync(self, target: npt.NDArray[np.floating[Any]]) -> None:
        values = np.asarray(target, dtype=np.float64).tolist()
        params = {
            "joint_positions": self._pb2.Value(float_array=self._pb2.FloatArray(values=values))
        }
        response = self._stub.Reset(
            self._pb2.ResetRequest(mode="target", params=params), timeout=120.0
        )
        self._check(response, "target reset")
        self._remember(response.observation)

    def move_gripper(self, width: float) -> None:
        if self._joints is None:
            self._refresh()
        assert self._joints is not None
        gripper_open = float(np.clip(width / self._cfg.gripper_max_width, 0.0, 1.0))
        action = [*self._joints.tolist(), 1.0 - gripper_open]
        self._step(action, "joint_position")
        self._gripper_open_command = gripper_open

    def move_cartesian_delta(
        self, position_delta: npt.NDArray[np.floating[Any]], gripper_open_delta: float
    ) -> None:
        if self._gripper_open is None:
            self._refresh()
        baseline = self._gripper_open_command
        if baseline is None:
            baseline = self._gripper_open
        assert baseline is not None
        target_open = float(np.clip(baseline + gripper_open_delta, 0.0, 1.0))
        xyz = np.asarray(position_delta, dtype=np.float64)
        if xyz.shape != (3,):
            raise ValueError(f"expected a 3-D Cartesian delta, got shape {xyz.shape}")
        self._step([*xyz.tolist(), 0.0, 0.0, 0.0, 1.0 - target_open], "cartesian_delta")
        self._gripper_open_command = target_open

    def disconnect(self) -> None:
        self._channel.close()


def robotenv_driver_factory(cfg: FrankaConfig) -> Driver:
    """Connect one configured RobotEnv endpoint."""
    return _RobotEnvDriver(cfg)


class BimanualRobotEnvEmbodiment(BimanualFrankaEmbodiment):
    """Y-frame Robotiq pair using franka-controller instead of direct FCI."""

    RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {
        "grpc": GRPC_INSTALL_COMMAND,
        "proto.robotenv_pb2": PROTO_INSTALL_COMMAND,
        "cv2": "pip install opencv-python-headless",
    }

    def __init__(
        self,
        config: BimanualFrankaConfig | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            kwargs.setdefault("home_pose", Y_FRAME_ROBOTIQ_HOME_POSE)
            kwargs.setdefault("rest_pose", kwargs["home_pose"])
        super().__init__(
            config,
            driver_factory=driver_factory or robotenv_driver_factory,
            **kwargs,
        )
        docs = (
            "Two Franka arms controlled by position-only Cartesian deltas in the shared "
            "Y-frame world coordinates. Use move_by with left_dx/left_dy/left_dz or "
            "right_dx/right_dy/right_dz in metres. Positive gripper deltas open and "
            "negative deltas close the named hand. End-effector rotations stay fixed. "
            "Move one arm at a time near objects or the shared workspace and inspect the "
            "cameras after each small motion. This profile requires y_frame_v1 with "
            "Robotiq grippers; RobotEnv converts the open-positive gripper convention."
        )
        if self._cfg.docs_extra.strip():
            docs += "\n\n" + self._cfg.docs_extra.strip()
        self.info = dataclasses.replace(
            self.info,
            name="franka_bimanual_robotenv",
            action_space=robotenv_action_box(),
            docs=docs,
        )

    def step(self, action: Action) -> StepResult:
        """Send bounded XYZ and gripper deltas to both RobotEnv services."""
        drivers = cast(Mapping[str, CartesianDriver], self._require_drivers())
        self.num_steps += 1
        command = np.asarray(action.data, dtype=np.float64)
        if command.ndim != 1 or command.shape != ROBOTENV_ACTION_LOW.shape:
            raise ValueError(
                f"expected an {ROBOTENV_ACTION_LOW.size}-D vector, got shape {command.shape}"
            )
        clamped = np.clip(command, ROBOTENV_ACTION_LOW, ROBOTENV_ACTION_HIGH)
        for index, side in enumerate(ARMS):
            part = clamped[index * 4 : (index + 1) * 4]
            drivers[side].move_cartesian_delta(part[:3], float(part[3]))
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
