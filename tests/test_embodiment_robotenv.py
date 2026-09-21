from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_robots.conformance import missing_runtime_requirements
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from inspect_robots_franka.config import FrankaConfig
from inspect_robots_franka.config_bimanual import BimanualFrankaConfig
from inspect_robots_franka.embodiment_robotenv import (
    GRPC_INSTALL_COMMAND,
    PROTO_INSTALL_COMMAND,
    ROBOTENV_ACTION_HIGH,
    ROBOTENV_ACTION_LOW,
    ROBOTENV_ACTION_SEMANTICS,
    ROBOTENV_DIM_LABELS,
    Y_FRAME_ROBOTIQ_HOME_POSE,
    BimanualRobotEnvEmbodiment,
    _RobotEnvDriver,
    robotenv_driver_factory,
)


class _Value:
    def __init__(self, *, scalar: float | None = None, array: list[float] | None = None) -> None:
        self.float_value = scalar or 0.0
        self.float_array = SimpleNamespace(values=[] if array is None else array)
        self._scalar = scalar is not None
        self._array = array is not None

    def HasField(self, name: str) -> bool:
        return self._scalar if name == "float_value" else self._array


def _observation(
    *, joints: list[float] | None = None, closed: float = 0.0, gripper: _Value | None = None
) -> dict[str, _Value]:
    return {
        "joint_positions": _Value(array=list(range(7)) if joints is None else joints),
        "gripper_position": gripper or _Value(scalar=closed),
    }


def _response(
    *, status: str = "SUCCESS", message: str = "", observation: Any = None
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        message=message,
        observation=_observation() if observation is None else observation,
    )


class _Stub:
    def __init__(self) -> None:
        self.supported = ["cartesian_delta", "joint_position"]
        self.frame_type = "y_frame_v1"
        self.gripper_type = "robotiq"
        self.health = "HEALTHY"
        self.health_message = "ready"
        self.reset_response = _response()
        self.step_response = _response()
        self.reset_requests: list[tuple[Any, float]] = []
        self.step_requests: list[tuple[Any, float]] = []

    def GetConfig(self, _request: Any, *, timeout: float) -> SimpleNamespace:
        assert timeout == 5.0
        return SimpleNamespace(
            supported_action_spaces=self.supported,
            frame_type=self.frame_type,
            gripper_type=self.gripper_type,
        )

    def HealthCheck(self, _request: Any, *, timeout: float) -> SimpleNamespace:
        assert timeout == 5.0
        return SimpleNamespace(status=self.health, message=self.health_message)

    def Reset(self, request: Any, *, timeout: float) -> SimpleNamespace:
        self.reset_requests.append((request, timeout))
        return self.reset_response

    def Step(self, request: Any, *, timeout: float) -> SimpleNamespace:
        self.step_requests.append((request, timeout))
        return self.step_response


class _Channel:
    def __init__(self, endpoint: str, stub: _Stub) -> None:
        self.endpoint = endpoint
        self.stub = stub
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Grpc:
    def __init__(self, stub: _Stub) -> None:
        self.stub = stub
        self.channel: _Channel | None = None

    def insecure_channel(self, endpoint: str) -> _Channel:
        self.channel = _Channel(endpoint, self.stub)
        return self.channel


class _Pb2:
    @staticmethod
    def GetConfigRequest() -> SimpleNamespace:
        return SimpleNamespace()

    @staticmethod
    def HealthCheckRequest() -> SimpleNamespace:
        return SimpleNamespace()

    @staticmethod
    def ResetRequest(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    @staticmethod
    def StepRequest(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    @staticmethod
    def FloatArray(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    @staticmethod
    def Value(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)


class _Pb2Grpc:
    @staticmethod
    def RobotEnvStub(channel: _Channel) -> _Stub:
        return channel.stub


def _driver(stub: _Stub | None = None) -> tuple[_RobotEnvDriver, _Stub, _Grpc]:
    actual_stub = stub or _Stub()
    grpc = _Grpc(actual_stub)
    driver = _RobotEnvDriver(
        FrankaConfig(hostname="localhost:50061"), modules=(grpc, _Pb2, _Pb2Grpc)
    )
    return driver, actual_stub, grpc


def test_driver_maps_joint_commands_and_inverts_gripper_polarity() -> None:
    driver, stub, grpc = _driver()
    home = np.arange(7, dtype=np.float64) / 10
    driver.move_joints_sync(home)
    request, timeout = stub.reset_requests[-1]
    assert request.mode == "target"
    assert request.params["joint_positions"].float_array.values == pytest.approx(home)
    assert timeout == 120.0
    assert driver.read_joints() == pytest.approx(np.arange(7))
    assert driver.read_gripper_width() == pytest.approx(0.08)

    driver.move_joints(home)
    request, timeout = stub.step_requests[-1]
    assert request.action == pytest.approx([*home, 0.0])
    assert request.action_space == "joint_position"
    assert request.gripper_action_space == "position"
    assert timeout == 5.0

    driver.move_gripper(0.0)
    assert stub.step_requests[-1][0].action[-1] == pytest.approx(1.0)
    driver.disconnect()
    assert grpc.channel is not None and grpc.channel.closed


def test_driver_maps_position_only_cartesian_deltas_and_accumulates_gripper() -> None:
    driver, stub, _ = _driver()
    driver.move_cartesian_delta(np.asarray([0.01, -0.02, 0.005]), -0.2)
    request, timeout = stub.step_requests[-1]
    assert request.action == pytest.approx([0.01, -0.02, 0.005, 0.0, 0.0, 0.0, 0.2])
    assert request.action_space == "cartesian_delta"
    assert request.gripper_action_space == "position"
    assert timeout == 5.0

    driver.move_cartesian_delta(np.zeros(3), 0.1)
    assert stub.step_requests[-1][0].action[-1] == pytest.approx(0.1)
    driver._gripper_open = 0.4
    driver._gripper_open_command = None
    driver.move_cartesian_delta(np.zeros(3), 0.1)
    assert stub.step_requests[-1][0].action[-1] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="expected a 3-D Cartesian delta"):
        driver.move_cartesian_delta(np.zeros(2), 0.0)


def test_driver_refreshes_state_and_accepts_single_value_arrays() -> None:
    stub = _Stub()
    stub.reset_response = _response(observation=_observation(gripper=_Value(array=[0.25])))
    driver, _, _ = _driver(stub)
    assert driver.read_joints() == pytest.approx(np.arange(7))
    driver._gripper_open = None
    assert driver.read_gripper_width() == pytest.approx(0.06)
    driver._joints = None
    driver.move_gripper(0.08)
    driver._gripper_open = None
    driver.move_joints(np.zeros(7))
    assert [request.mode for request, _timeout in stub.reset_requests] == [
        "current",
        "current",
        "current",
        "current",
    ]


@pytest.mark.parametrize("operation", ["refresh", "step", "reset"])
def test_driver_surfaces_robotenv_status_failures(operation: str) -> None:
    stub = _Stub()
    driver, _, _ = _driver(stub)
    failure = _response(status="FAILED", message="controller fault")
    if operation == "refresh":
        stub.reset_response = failure
        with pytest.raises(RuntimeError, match="current-state read failed"):
            driver.read_joints()
    elif operation == "step":
        stub.step_response = failure
        driver._gripper_open = 1.0
        with pytest.raises(RuntimeError, match="step failed"):
            driver.move_joints(np.zeros(7))
    else:
        stub.reset_response = failure
        with pytest.raises(RuntimeError, match="target reset failed"):
            driver.move_joints_sync(np.zeros(7))


def test_driver_rejects_incompatible_or_unhealthy_services() -> None:
    stub = _Stub()
    stub.supported = []
    with pytest.raises(RuntimeError, match="does not support cartesian_delta, joint_position"):
        _driver(stub)
    stub.supported = ["joint_position"]
    with pytest.raises(RuntimeError, match="does not support cartesian_delta"):
        _driver(stub)
    stub.supported = ["cartesian_delta", "joint_position"]
    stub.frame_type = "plane_frame_v1"
    with pytest.raises(RuntimeError, match=r"plane_frame_v1 \+ robotiq"):
        _driver(stub)
    stub.frame_type = "y_frame_v1"
    stub.gripper_type = "franka_hand"
    with pytest.raises(RuntimeError, match=r"y_frame_v1 \+ franka_hand"):
        _driver(stub)
    stub.gripper_type = "robotiq"
    stub.health = "DEGRADED"
    with pytest.raises(RuntimeError, match="is DEGRADED: ready"):
        _driver(stub)
    with pytest.raises(ValueError, match="endpoint is required"):
        _RobotEnvDriver(FrankaConfig(), modules=(_Grpc(stub), _Pb2, _Pb2Grpc))


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (_observation(joints=[0.0] * 6), r"joints of shape \(6,\)"),
        (_observation(gripper=_Value(array=[0.0, 1.0])), "one normalized value"),
    ],
)
def test_driver_rejects_invalid_observations(observation: dict[str, _Value], message: str) -> None:
    stub = _Stub()
    stub.reset_response = _response(observation=observation)
    driver, _, _ = _driver(stub)
    with pytest.raises(ValueError, match=message):
        driver.read_joints()


def test_factory_and_registered_embodiment_remain_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _Stub()
    grpc = _Grpc(stub)
    monkeypatch.setattr(
        "inspect_robots_franka.embodiment_robotenv._load_robotenv",
        lambda: (grpc, _Pb2, _Pb2Grpc),
    )
    assert isinstance(
        robotenv_driver_factory(FrankaConfig(hostname="localhost:50061")),
        _RobotEnvDriver,
    )

    calls: list[str | None] = []

    def factory(cfg: FrankaConfig) -> Any:
        calls.append(cfg.hostname)
        return SimpleNamespace()

    embodiment = BimanualRobotEnvEmbodiment(
        driver_factory=factory,
        left_hostname="localhost:50061",
        right_hostname="localhost:50063",
        docs_extra="Rig note.",
    )
    assert calls == []
    assert embodiment.info.name == "franka_bimanual_robotenv"
    assert embodiment.info.action_space.shape == (8,)
    assert embodiment.info.action_space.low == pytest.approx(ROBOTENV_ACTION_LOW)
    assert embodiment.info.action_space.high == pytest.approx(ROBOTENV_ACTION_HIGH)
    assert embodiment.info.action_space.semantics is ROBOTENV_ACTION_SEMANTICS
    assert ROBOTENV_ACTION_SEMANTICS.control_mode == "eef_delta_pos"
    assert ROBOTENV_ACTION_SEMANTICS.frame == "world"
    assert ROBOTENV_ACTION_SEMANTICS.dim_labels == ROBOTENV_DIM_LABELS
    assert "y_frame_v1" in embodiment.info.docs
    assert "Rig note." in embodiment.info.docs
    assert embodiment._cfg.home_pose == Y_FRAME_ROBOTIQ_HOME_POSE
    assert embodiment._cfg.rest_pose == Y_FRAME_ROBOTIQ_HOME_POSE

    explicit = BimanualFrankaConfig(rest_pose=None)
    configured = BimanualRobotEnvEmbodiment(explicit, driver_factory=factory)
    assert configured._cfg is explicit
    assert configured._cfg.rest_pose is None


class _CartesianFakeDriver:
    def __init__(self) -> None:
        self.joints = np.zeros(7)
        self.width = 0.08
        self.cartesian_commands: list[tuple[np.ndarray, float]] = []

    def read_joints(self) -> np.ndarray:
        return self.joints.copy()

    def read_gripper_width(self) -> float:
        return self.width

    def move_joints(self, target: np.ndarray) -> None:
        self.joints = np.asarray(target).copy()

    def move_joints_sync(self, target: np.ndarray) -> None:
        self.joints = np.asarray(target).copy()

    def move_gripper(self, width: float) -> None:
        self.width = width

    def move_cartesian_delta(self, position_delta: np.ndarray, gripper_delta: float) -> None:
        self.cartesian_commands.append((np.asarray(position_delta).copy(), gripper_delta))

    def disconnect(self) -> None:
        pass


def test_registered_embodiment_clamps_and_sends_cartesian_deltas() -> None:
    left = _CartesianFakeDriver()
    right = _CartesianFakeDriver()
    drivers = iter((left, right))
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    embodiment = BimanualRobotEnvEmbodiment(
        left_hostname="localhost:50061",
        right_hostname="localhost:50063",
        unattended=True,
        driver_factory=lambda _cfg: next(drivers),
        camera_reader=lambda: {
            "exterior_cam": image,
            "left_wrist_cam": image,
            "right_wrist_cam": image,
        },
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
    )
    embodiment.reset(Scene(id="s", instruction="test Cartesian motion"))
    result = embodiment.step(Action(data=np.asarray([1.0, -1.0, 0.01, -1.0] * 2)))

    for driver in (left, right):
        xyz, gripper = driver.cartesian_commands[-1]
        assert xyz == pytest.approx([0.02, -0.02, 0.01])
        assert gripper == pytest.approx(-0.2)
    assert result.terminated is False
    with pytest.raises(ValueError, match="expected an 8-D vector"):
        embodiment.step(Action(data=np.zeros(7)))

    embodiment._cfg = dataclasses.replace(embodiment._cfg, unattended=False)
    embodiment._poll_end = lambda: True
    embodiment._operator = SimpleNamespace(confirm_success=lambda: True)
    result = embodiment.step(Action(data=np.zeros(8)))
    assert result.terminated is True
    assert result.termination_reason == "success"


def test_y_frame_robotiq_home_pose_matches_rci() -> None:
    assert Y_FRAME_ROBOTIQ_HOME_POSE == (
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


def test_runtime_requirements_name_grpc_proto_and_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    assert missing_runtime_requirements(BimanualRobotEnvEmbodiment) == {
        "grpc": GRPC_INSTALL_COMMAND,
        "proto.robotenv_pb2": PROTO_INSTALL_COMMAND,
        "cv2": "pip install opencv-python-headless",
    }
