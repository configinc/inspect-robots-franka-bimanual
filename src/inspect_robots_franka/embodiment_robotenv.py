"""Bimanual Franka embodiment backed by franka-controller RobotEnv gRPC."""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt

from inspect_robots_franka.config import FrankaConfig
from inspect_robots_franka.config_bimanual import BimanualFrankaConfig
from inspect_robots_franka.embodiment import Driver, DriverFactory
from inspect_robots_franka.embodiment_bimanual import BimanualFrankaEmbodiment

GRPC_INSTALL_COMMAND = 'pip install "inspect-robots-franka[robotenv]"'
PROTO_INSTALL_COMMAND = "export PYTHONPATH=/home/ubuntu/franka-controller:$PYTHONPATH"
Y_FRAME_ROBOTIQ_HOME_POSE: tuple[float, ...] = (
    0.4755530,
    0.2318635,
    -0.5844381,
    -2.2414968,
    0.8894346,
    2.9037103,
    0.4044128,
    1.0,
    -0.1042003,
    0.0628366,
    0.2630598,
    -2.3995060,
    0.1996380,
    2.9613211,
    -1.2133284,
    1.0,
)


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

        robot_cfg = self._stub.GetConfig(self._pb2.GetConfigRequest(), timeout=5.0)
        if "joint_position" not in robot_cfg.supported_action_spaces:
            raise RuntimeError(f"RobotEnv at {cfg.hostname} does not support joint_position")
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

    def _step(self, joints: npt.NDArray[np.floating[Any]], gripper_open: float) -> None:
        action = [*np.asarray(joints, dtype=np.float64).tolist(), 1.0 - gripper_open]
        response = self._stub.Step(
            self._pb2.StepRequest(
                action=action,
                action_space="joint_position",
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
        self._step(target, self._gripper_open)

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
        self._step(self._joints, gripper_open)

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
        self.info = dataclasses.replace(
            self.info,
            name="franka_bimanual_robotenv",
            docs=(self.info.docs or "")
            + "\n\nThis profile requires a y_frame_v1 pair with Robotiq grippers. Robot "
            "transport uses franka-controller RobotEnv gRPC, and endpoint commands are "
            "converted to its closed-positive gripper convention.",
        )
