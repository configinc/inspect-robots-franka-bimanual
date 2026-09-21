"""API-free policy that holds a bimanual Franka at its observed pose."""

from __future__ import annotations

import numpy as np
from inspect_robots.embodiment import EmbodimentInfo
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
        self._messages: list[dict[str, str]] = []
        self._delta_cursor = 0
        self._displacement = False

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Adopt the embodiment contract so zero deltas hold Cartesian rigs still."""
        self.info = PolicyInfo(
            name="bimanual_hold",
            action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,
        )
        semantics = embodiment_info.action_space.semantics
        mode = semantics.control_mode if semantics is not None else None
        if mode not in {"joint_pos", "eef_delta_pos"}:
            raise ValueError(f"bimanual_hold cannot safely hold control mode {mode!r}")
        self._displacement = mode == "eef_delta_pos"

    def reset(self, scene: Scene) -> None:
        """Reset the diagnostic inference counter."""
        self.num_inferences = 0
        self._messages = [{"role": "user", "content": scene.instruction or ""}]
        self._delta_cursor = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Return one action equal to the latest observed robot state."""
        self.num_inferences += 1
        if self.num_inferences == 1:
            detail = (
                f"sending zero {self.info.action_space.dim}-D Cartesian deltas"
                if self._displacement
                else "keeping the observed 16-D pose unchanged"
            )
            self._messages.append(
                {
                    "role": "assistant",
                    "content": f"API-free hold: {detail}.",
                }
            )
        data = (
            np.zeros(self.info.action_space.dim, dtype=np.float64)
            if self._displacement
            else validate_dim(observation.state[STATE_KEY]).copy()
        )
        return ActionChunk(actions=[Action(data=data)])

    def transcript(self) -> list[dict[str, str]]:
        """Return the small audit transcript used by the HTML video renderer."""
        return [dict(message) for message in self._messages]

    def transcript_delta(self) -> list[dict[str, str]] | None:
        """Return new audit messages for the live HTML viewer."""
        messages = [dict(message) for message in self._messages[self._delta_cursor :]]
        self._delta_cursor = len(self._messages)
        return messages or None
