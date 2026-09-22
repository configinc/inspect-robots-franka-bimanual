from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from inspect_robots.conformance import device_slots, missing_runtime_requirements
from inspect_robots.embodiment import SELF_PACED
from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.task import TaskEnvelope
from inspect_robots.types import Action

from inspect_robots_franka.config import FrankaConfig
from inspect_robots_franka.config_bimanual import DEFAULT_HOME_POSE, BimanualFrankaConfig
from inspect_robots_franka.embodiment import Driver
from inspect_robots_franka.embodiment_bimanual import BimanualFrankaEmbodiment
from inspect_robots_franka.operator import OperatorIO

LEFT = "left.local"
RIGHT = "right.local"


class _FakeDriver:
    def __init__(self, side: str, events: list[str] | None = None) -> None:
        self.side = side
        self.joints = np.zeros(7)
        self.width = 0.02
        self.async_joints: list[np.ndarray] = []
        self.sync_joints: list[np.ndarray] = []
        self.gripper_commands: list[float] = []
        self.disconnect_calls = 0
        self.events = events
        self.fail_sync = False
        self.fail_disconnect = False

    def _log(self, event: str) -> None:
        if self.events is not None:
            self.events.append(f"{self.side}:{event}")

    def read_joints(self) -> np.ndarray:
        return self.joints.copy()

    def read_gripper_width(self) -> float:
        return self.width

    def move_joints(self, target: np.ndarray) -> None:
        self.async_joints.append(np.asarray(target).copy())
        self.joints = np.asarray(target).copy()
        self._log("move_joints")

    def move_joints_sync(self, target: np.ndarray) -> None:
        if self.fail_sync:
            raise RuntimeError(f"{self.side} park failed")
        self.sync_joints.append(np.asarray(target).copy())
        self.joints = np.asarray(target).copy()
        self._log("move_joints_sync")

    def move_gripper(self, width: float) -> None:
        self.gripper_commands.append(width)
        self.width = width
        self._log("move_gripper")

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._log("disconnect")
        if self.fail_disconnect:
            raise RuntimeError(f"{self.side} disconnect failed")


class _Rig:
    """Two fake drivers handed out by hostname, with a recording factory."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.left = _FakeDriver("left", events)
        self.right = _FakeDriver("right", events)
        self.factory_calls: list[str | None] = []
        self.fail_for: str | None = None

    def factory(self, cfg: FrankaConfig) -> Driver:
        self.factory_calls.append(cfg.hostname)
        if self.events is not None:
            self.events.append(f"factory:{cfg.hostname}")
        if cfg.hostname == self.fail_for:
            raise RuntimeError(f"cannot reach {cfg.hostname}")
        return {LEFT: self.left, RIGHT: self.right}[cfg.hostname or ""]


def _camera_reader() -> dict[str, np.ndarray]:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    return {"exterior_cam": image, "left_wrist_cam": image, "right_wrist_cam": image}


def _scene() -> Scene:
    return Scene(id="s", instruction="hand the cube from left to right")


def _config(**overrides: Any) -> BimanualFrankaConfig:
    fields: dict[str, Any] = {"left_hostname": LEFT, "right_hostname": RIGHT, "unattended": True}
    fields.update(overrides)
    return BimanualFrankaConfig(**fields)


def _embodiment(
    rig: _Rig,
    cfg: BimanualFrankaConfig | None = None,
    **kwargs: Any,
) -> BimanualFrankaEmbodiment:
    return BimanualFrankaEmbodiment(
        cfg or _config(),
        driver_factory=rig.factory,
        camera_reader=_camera_reader,
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
        **kwargs,
    )


def _home_with(left_gripper: float | None = None, right_gripper: float | None = None) -> np.ndarray:
    pose = np.asarray(DEFAULT_HOME_POSE, dtype=np.float64)
    if left_gripper is not None:
        pose[7] = left_gripper
    if right_gripper is not None:
        pose[15] = right_gripper
    return pose


def test_init_is_inert_and_declares_the_sixteen_d_contract() -> None:
    rig = _Rig()
    embodiment = BimanualFrankaEmbodiment(driver_factory=rig.factory)
    assert rig.factory_calls == []
    assert embodiment.info.name == "franka_bimanual"
    assert embodiment.info.action_space.dim == 16
    assert embodiment.info.control_hz == 15.0
    assert embodiment.info.capabilities == frozenset({SELF_PACED})
    assert embodiment.info.observation_space.camera_names == frozenset(_camera_reader())


def test_flat_kwargs_build_the_config() -> None:
    embodiment = BimanualFrankaEmbodiment(
        left_hostname=LEFT, right_hostname=RIGHT, control_hz="10", unattended="true"
    )
    assert embodiment.info.control_hz == 10.0
    assert embodiment._cfg.unattended is True


def test_reset_connects_both_arms_lazily_and_homes_them_one_at_a_time() -> None:
    events: list[str] = []
    rig = _Rig(events)
    embodiment = _embodiment(rig)
    observation = embodiment.reset(_scene())
    assert rig.factory_calls == [LEFT, RIGHT]
    assert events == [
        f"factory:{LEFT}",
        f"factory:{RIGHT}",
        "left:move_joints_sync",
        "left:move_gripper",
        "right:move_joints_sync",
        "right:move_gripper",
    ]
    assert rig.left.sync_joints[0] == pytest.approx(DEFAULT_HOME_POSE[:7])
    assert rig.right.sync_joints[0] == pytest.approx(DEFAULT_HOME_POSE[8:15])
    assert rig.left.gripper_commands == rig.right.gripper_commands == pytest.approx([0.08])
    state = observation.state["joint_pos"]
    assert state.shape == (16,)
    assert state[7] == state[15] == pytest.approx(1.0)
    assert state[:7] == pytest.approx(DEFAULT_HOME_POSE[:7])
    assert observation.instruction == "hand the cube from left to right"
    assert observation.state_time == 0.0
    assert set(observation.image_times) == {"exterior_cam", "left_wrist_cam", "right_wrist_cam"}


def test_operator_stand_clear_and_scene_ready_wrap_both_homings() -> None:
    events: list[str] = []
    rig = _Rig(events)
    prompts: list[str] = []

    def prompt(text: str) -> str:
        prompts.append(text)
        events.append("stand_clear" if "Stand clear" in text else "scene_ready")
        return ""

    embodiment = _embodiment(
        rig,
        _config(unattended=False),
        operator=OperatorIO(input_fn=prompt, output_fn=lambda _line: None),
        poll_end=lambda: False,
    )
    embodiment.reset(_scene())
    assert events[2:] == [
        "stand_clear",
        "left:move_joints_sync",
        "left:move_gripper",
        "right:move_joints_sync",
        "right:move_gripper",
        "scene_ready",
    ]
    assert prompts[0].startswith("Both arms will move to the home pose.")


def test_step_hard_clamps_each_half_and_routes_it_to_its_own_arm() -> None:
    rig = _Rig()
    sleeps: list[float] = []
    cfg = _config()
    embodiment = BimanualFrankaEmbodiment(
        cfg,
        driver_factory=rig.factory,
        camera_reader=_camera_reader,
        sleep_fn=sleeps.append,
        clock=lambda: 0.0,
    )
    embodiment.reset(_scene())
    action = np.concatenate((np.full(8, 100.0), np.full(8, -100.0)))
    result = embodiment.step(Action(data=action))
    assert rig.left.async_joints[-1] == pytest.approx(cfg.high[:7])
    assert rig.right.async_joints[-1] == pytest.approx(cfg.low[8:15])
    assert rig.left.gripper_commands[-1] == pytest.approx(cfg.gripper_max_width)
    assert rig.right.gripper_commands[-1] == pytest.approx(0.0)
    assert sleeps == pytest.approx([1.0 / 15.0])
    assert result.terminated is False
    assert embodiment.num_steps == 1
    assert result.observation.state["joint_pos"][8:15] == pytest.approx(cfg.low[8:15])


def test_pacing_never_sleeps_negative_time() -> None:
    rig = _Rig()
    times = iter([0.0, 0.0, 1.0, 1.0, 1.0])
    sleeps: list[float] = []
    embodiment = BimanualFrankaEmbodiment(
        _config(),
        driver_factory=rig.factory,
        camera_reader=_camera_reader,
        sleep_fn=sleeps.append,
        clock=times.__next__,
    )
    embodiment.reset(_scene())
    embodiment.step(Action(data=_home_with()))
    assert sleeps == [0.0]


def test_gripper_deadband_is_tracked_per_arm() -> None:
    rig = _Rig()
    embodiment = _embodiment(rig)
    embodiment.reset(_scene())
    rig.left.gripper_commands.clear()
    rig.right.gripper_commands.clear()
    embodiment.step(Action(data=_home_with()))
    assert rig.left.gripper_commands == rig.right.gripper_commands == []
    embodiment.step(Action(data=_home_with(right_gripper=0.25)))
    assert rig.left.gripper_commands == []
    assert rig.right.gripper_commands == pytest.approx([0.02])
    embodiment.step(Action(data=_home_with(left_gripper=0.5, right_gripper=0.25)))
    assert rig.left.gripper_commands == pytest.approx([0.04])
    assert rig.right.gripper_commands == pytest.approx([0.02])
    embodiment.step(Action(data=_home_with(left_gripper=0.45, right_gripper=0.2)))
    assert len(rig.left.gripper_commands) == len(rig.right.gripper_commands) == 1


@pytest.mark.parametrize(
    "devices",
    [
        (None, None, None),
        ("/dev/video0", None, None),
        ("/dev/video0", "/dev/video1", None),
        (None, "/dev/video1", "/dev/video2"),
    ],
)
def test_missing_or_partial_camera_config_fails_before_connect(
    devices: tuple[str | None, str | None, str | None],
) -> None:
    rig = _Rig()
    cfg = _config(
        exterior_cam_device=devices[0],
        left_wrist_cam_device=devices[1],
        right_wrist_cam_device=devices[2],
    )
    embodiment = BimanualFrankaEmbodiment(cfg, driver_factory=rig.factory)
    with pytest.raises(ConfigError, match="require exterior_cam_device, left_wrist_cam_device"):
        embodiment.reset(_scene())
    assert rig.factory_calls == []


@pytest.mark.parametrize(
    ("hostnames", "message"),
    [
        ((LEFT, None), "right_hostname required"),
        ((None, RIGHT), "left_hostname required"),
        ((None, None), "left_hostname and right_hostname required"),
    ],
)
def test_missing_hostnames_are_named_after_the_camera_check(
    hostnames: tuple[str | None, str | None], message: str
) -> None:
    rig = _Rig()
    cfg = BimanualFrankaConfig(left_hostname=hostnames[0], right_hostname=hostnames[1])
    embodiment = BimanualFrankaEmbodiment(
        cfg, driver_factory=rig.factory, camera_reader=_camera_reader
    )
    with pytest.raises(ConfigError, match=message):
        embodiment.reset(_scene())
    assert rig.factory_calls == []


def test_injected_reader_wins_over_partial_devices_and_bad_reader_is_rejected() -> None:
    rig = _Rig()
    cfg = _config(exterior_cam_device="ignored")
    embodiment = _embodiment(rig, cfg)
    assert embodiment.reset(_scene()).images.keys() == _camera_reader().keys()

    invalid_reader = BimanualFrankaEmbodiment(
        _config(),
        driver_factory=rig.factory,
        camera_reader="not callable",  # type: ignore[arg-type]
    )
    with pytest.raises(ConfigError, match="must be callable"):
        invalid_reader.reset(_scene())


def test_three_device_config_selects_builtin_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _Rig()
    calls: list[tuple[dict[str, object], int, int]] = []

    def builder(
        devices: dict[str, object], *, width: int, height: int
    ) -> Callable[[], dict[str, np.ndarray]]:
        calls.append((dict(devices), width, height))
        return _camera_reader

    monkeypatch.setattr("inspect_robots_franka.embodiment_bimanual.opencv_camera_reader", builder)
    cfg = _config(
        exterior_cam_device=0,
        left_wrist_cam_device=1,
        right_wrist_cam_device="/dev/v4l/by-id/right",
        cam_width=320,
        cam_height=240,
    )
    embodiment = BimanualFrankaEmbodiment(
        cfg, driver_factory=rig.factory, sleep_fn=lambda _delay: None, clock=lambda: 0.0
    )
    embodiment.reset(_scene())
    assert calls == [
        (
            {"exterior_cam": 0, "left_wrist_cam": 1, "right_wrist_cam": "/dev/v4l/by-id/right"},
            320,
            240,
        )
    ]


@pytest.mark.parametrize(("verdict", "reason"), [("yes", "success"), ("no", "failure")])
def test_operator_verdict_uses_termination_reason(verdict: str, reason: str) -> None:
    rig = _Rig()
    answers = iter(["", "", verdict])
    embodiment = _embodiment(
        rig,
        _config(unattended=False),
        operator=OperatorIO(input_fn=lambda _prompt: next(answers), output_fn=lambda _l: None),
        poll_end=lambda: True,
    )
    embodiment.reset(_scene())
    result = embodiment.step(Action(data=_home_with()))
    assert result.terminated is True
    assert result.termination_reason == reason
    assert result.info == {"operator_confirmed": verdict == "yes"}


def test_unattended_skips_poll_and_prompts() -> None:
    rig = _Rig()
    polled = 0

    def poll() -> bool:
        nonlocal polled
        polled += 1
        return True

    embodiment = _embodiment(rig, poll_end=poll)
    embodiment.reset(_scene())
    result = embodiment.step(Action(data=_home_with()))
    assert result.terminated is False
    assert polled == 0


def test_close_parks_both_arms_arm_only_and_is_idempotent() -> None:
    events: list[str] = []
    rig = _Rig(events)
    rest = tuple(_home_with(left_gripper=0.25, right_gripper=0.25))
    embodiment = _embodiment(rig, _config(rest_pose=rest))
    embodiment.reset(_scene())
    events.clear()
    embodiment.close()
    embodiment.close()
    assert events == [
        "left:move_joints_sync",
        "right:move_joints_sync",
        "left:disconnect",
        "right:disconnect",
    ]
    assert rig.left.sync_joints[-1] == pytest.approx(rest[:7])
    assert rig.right.sync_joints[-1] == pytest.approx(rest[8:15])
    assert rig.left.disconnect_calls == rig.right.disconnect_calls == 1


def test_close_disconnects_every_arm_when_parking_or_a_disconnect_fails() -> None:
    rig = _Rig()
    embodiment = _embodiment(rig, _config(rest_pose=DEFAULT_HOME_POSE))
    embodiment.reset(_scene())
    rig.left.sync_joints.clear()
    rig.right.sync_joints.clear()
    rig.left.fail_sync = True
    with pytest.raises(RuntimeError, match="left park failed"):
        embodiment.close()
    assert rig.right.sync_joints == []
    assert rig.left.disconnect_calls == rig.right.disconnect_calls == 1
    embodiment.close()
    assert rig.left.disconnect_calls == 1

    rig2 = _Rig()
    embodiment2 = _embodiment(rig2)
    embodiment2.reset(_scene())
    rig2.left.fail_disconnect = True
    rig2.right.fail_disconnect = True
    with pytest.raises(RuntimeError, match="left disconnect failed"):
        embodiment2.close()
    assert rig2.left.disconnect_calls == rig2.right.disconnect_calls == 1
    with pytest.raises(RuntimeError, match="before reset"):
        embodiment2.step(Action(data=_home_with()))
    embodiment2.close()
    assert rig2.right.disconnect_calls == 1


def test_partial_connection_parks_only_the_connected_arm_at_close() -> None:
    rig = _Rig()
    rig.fail_for = RIGHT
    embodiment = _embodiment(rig, _config(rest_pose=DEFAULT_HOME_POSE))
    with pytest.raises(RuntimeError, match=r"cannot reach right\.local"):
        embodiment.reset(_scene())
    assert rig.factory_calls == [LEFT, RIGHT]
    with pytest.raises(RuntimeError, match="before reset"):
        embodiment.step(Action(data=_home_with()))
    embodiment.close()
    assert len(rig.left.sync_joints) == 1
    assert rig.right.sync_joints == []
    assert rig.left.disconnect_calls == 1
    assert rig.right.disconnect_calls == 0


def test_partial_connection_is_resumed_by_the_next_reset() -> None:
    rig = _Rig()
    rig.fail_for = RIGHT
    embodiment = _embodiment(rig)
    with pytest.raises(RuntimeError, match="cannot reach"):
        embodiment.reset(_scene())
    rig.fail_for = None
    embodiment.reset(_scene())
    assert rig.factory_calls == [LEFT, RIGHT, RIGHT]
    assert len(rig.left.sync_joints) == 1
    assert len(rig.right.sync_joints) == 1
    embodiment.close()
    assert rig.left.disconnect_calls == rig.right.disconnect_calls == 1


def test_step_before_reset_and_bad_driver_joint_shape_name_the_arm() -> None:
    embodiment = _embodiment(_Rig())
    with pytest.raises(RuntimeError, match="before reset"):
        embodiment.step(Action(data=np.zeros(16)))
    rig = _Rig()
    embodiment = _embodiment(rig)
    embodiment.reset(_scene())
    rig.right.joints = np.zeros(6)
    with pytest.raises(ValueError, match="right driver returned joints"):
        embodiment._observe("instruction")
    with pytest.raises(ValueError, match="expected a 16-D vector"):
        embodiment.step(Action(data=np.zeros(8)))


def test_bind_task_runtime_requirements_and_device_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    embodiment = BimanualFrankaEmbodiment()
    embodiment.bind_task(TaskEnvelope(name="task", max_steps=42))
    assert embodiment._bound_max_steps == 42
    slots = device_slots(BimanualFrankaEmbodiment)
    assert [slot.arg for slot in slots] == [
        "exterior_cam_device",
        "left_wrist_cam_device",
        "right_wrist_cam_device",
    ]
    assert {slot.kind for slot in slots} == {"v4l2"}
    assert {slot.group for slot in slots} == {"cameras"}
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    missing = missing_runtime_requirements(BimanualFrankaEmbodiment)
    assert missing == {
        "franky": "pip install franky-control",
        "cv2": "pip install opencv-python-headless",
    }


def test_bound_task_horizon_appears_in_operator_status() -> None:
    rig = _Rig()
    output: list[str] = []
    embodiment = _embodiment(
        rig,
        _config(unattended=False),
        operator=OperatorIO(input_fn=lambda _prompt: "", output_fn=output.append),
        poll_end=lambda: False,
    )
    embodiment.bind_task(TaskEnvelope(name="task", max_steps=30))
    embodiment.reset(_scene())
    assert output == ["Running: press Enter to end the episode, then y/N to score. Max 2s."]


def test_close_without_connection_clears_bound_horizon() -> None:
    embodiment = BimanualFrankaEmbodiment()
    embodiment.bind_task(TaskEnvelope(name="task", max_steps=3))
    embodiment.close()
    assert embodiment._bound_max_steps is None
