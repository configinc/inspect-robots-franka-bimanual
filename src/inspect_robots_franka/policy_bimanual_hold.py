"""API-free policy that holds a bimanual Franka at its observed pose."""

from __future__ import annotations

from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_franka.config_bimanual import action_box, observation_space
from inspect_robots_franka.packing_bimanual import STATE_KEY, validate_dim


class BimanualHoldPolicy:
    """Echo the observed 16-D state to exercise rollout recording without an API."""

    def __init__(self) -> None:
        self.info = PolicyInfo(
            name="bimanual_hold",
            action_space=action_box(),
            observation_space=observation_space(),
        )
        self.config = PolicyConfig(action_horizon=1, replan_interval=1)
        self.num_inferences = 0

    def reset(self, scene: Scene) -> None:
        """Reset the diagnostic inference counter."""
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Return one action equal to the latest observed robot state."""
        self.num_inferences += 1
        state = validate_dim(observation.state[STATE_KEY]).copy()
        return ActionChunk(actions=[Action(data=state)])
