from __future__ import annotations

import numpy as np
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

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
