from __future__ import annotations

import numpy as np
from inspect_robots import eval as robots_eval
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_franka.config import FrankaConfig
from inspect_robots_franka.config_bimanual import (
    BimanualFrankaConfig,
    action_box,
    observation_space,
)
from inspect_robots_franka.embodiment_bimanual import BimanualFrankaEmbodiment
from inspect_robots_franka.operator import OperatorIO


class _Driver:
    def __init__(self) -> None:
        self.joints = np.zeros(7)
        self.width = 0.08

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

    def disconnect(self) -> None:
        return None


class _HoldStillPolicy:
    """Emit the observed 16-D joint state as the next absolute target."""

    config = PolicyConfig()

    def __init__(self) -> None:
        self.info = PolicyInfo(
            name="hold-still", action_space=action_box(), observation_space=observation_space()
        )
        self.seen: list[np.ndarray] = []

    def reset(self, scene: object) -> None:
        self.seen.clear()

    def act(self, observation: Observation) -> ActionChunk:
        state = np.asarray(observation.state["joint_pos"], dtype=np.float64)
        self.seen.append(state)
        return ActionChunk(actions=[Action(data=state)])


def _cameras() -> dict[str, np.ndarray]:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {"exterior_cam": image, "left_wrist_cam": image, "right_wrist_cam": image}


def test_full_eval_drives_both_arms_and_propagates_success() -> None:
    drivers = {"left.local": _Driver(), "right.local": _Driver()}

    def factory(cfg: FrankaConfig) -> _Driver:
        assert cfg.hostname is not None
        return drivers[cfg.hostname]

    policy = _HoldStillPolicy()
    embodiment = BimanualFrankaEmbodiment(
        BimanualFrankaConfig(left_hostname="left.local", right_hostname="right.local"),
        driver_factory=factory,
        camera_reader=_cameras,
        operator=OperatorIO(input_fn=lambda _prompt: "yes"),
        poll_end=lambda: True,
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
    )
    logs = robots_eval("cubepick-reach", policy, embodiment, sinks=[], seed=0)
    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.results.metrics["success_at_end"] == 1.0
    assert embodiment.num_steps == 1
    assert policy.seen[0].shape == (16,)
    assert drivers["left.local"].joints.shape == drivers["right.local"].joints.shape == (7,)
