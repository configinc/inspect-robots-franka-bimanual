from __future__ import annotations

import numpy as np
import pytest
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.types import Observation

from inspect_robots_franka.embodiment_robotenv import BimanualRobotEnvEmbodiment
from inspect_robots_franka.packing_bimanual import STATE_KEY, TOTAL_DIM
from inspect_robots_franka.policy_bimanual_hold import BimanualHoldPolicy


def test_bimanual_hold_echoes_observed_state_without_images_or_api() -> None:
    policy = BimanualHoldPolicy()
    state = np.arange(TOTAL_DIM, dtype=np.float64) / 10

    policy.reset(Scene(id="recording", instruction="record this rollout"))
    chunk = policy.act(Observation(images={}, state={STATE_KEY: state}))

    assert policy.info.name == "bimanual_hold"
    assert policy.info.action_space.dim == TOTAL_DIM
    assert policy.info.observation_space.camera_names == {
        "exterior_cam",
        "left_wrist_cam",
        "right_wrist_cam",
    }
    assert policy.num_inferences == 1
    assert np.array_equal(chunk.actions[0].data, state)
    assert chunk.actions[0].data is not state
    assert policy.transcript_delta() == [
        {"role": "user", "content": "record this rollout"},
        {
            "role": "assistant",
            "content": "API-free hold: keeping the observed 16-D pose unchanged.",
        },
    ]
    assert policy.transcript_delta() is None
    assert policy.transcript()[-1]["role"] == "assistant"

    policy.act(Observation(images={}, state={STATE_KEY: state}))
    assert policy.num_inferences == 2
    assert len(policy.transcript()) == 2


def test_bimanual_hold_binds_to_cartesian_delta_embodiment_with_zero_motion() -> None:
    policy = BimanualHoldPolicy()
    policy.bind(BimanualRobotEnvEmbodiment().info)
    policy.reset(Scene(id="recording", instruction="hold still"))

    chunk = policy.act(Observation(images={}, state={STATE_KEY: np.zeros(TOTAL_DIM)}))

    assert policy.info.action_space.semantics is not None
    assert policy.info.action_space.semantics.control_mode == "eef_delta_pos"
    assert np.array_equal(chunk.actions[0].data, np.zeros(8))
    assert policy.transcript()[-1]["content"] == (
        "API-free hold: sending zero 8-D Cartesian deltas."
    )


def test_bimanual_hold_rejects_unsafe_unrecognized_control_contracts() -> None:
    policy = BimanualHoldPolicy()
    info = BimanualRobotEnvEmbodiment().info
    unsupported = EmbodimentInfo(
        name=info.name,
        action_space=Box(shape=(8,), semantics=ActionSemantics(control_mode="eef_abs_pose")),
        observation_space=info.observation_space,
    )
    with pytest.raises(ValueError, match="cannot safely hold control mode 'eef_abs_pose'"):
        policy.bind(unsupported)
