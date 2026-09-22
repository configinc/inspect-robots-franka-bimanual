from __future__ import annotations

from typing import Any

from inspect_robots.compat import check_compatibility
from inspect_robots.conformance import assert_embodiment_conformant, check_embodiment
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import resolve
from inspect_robots.spaces import Box

from inspect_robots_franka.config_bimanual import action_box, observation_space
from inspect_robots_franka.embodiment_bimanual import BimanualFrankaEmbodiment
from inspect_robots_franka.policy import OpenpiPolicy


class _Policy:
    config = PolicyConfig()

    def __init__(self, info: PolicyInfo) -> None:
        self.info = info

    def reset(self, scene: object) -> None:
        return None

    def act(self, observation: object) -> Any:
        raise AssertionError("not called")


class _BindingPolicy(_Policy):
    """Mimic an LLM agent policy that adopts the embodiment contract at bind time."""

    def __init__(self) -> None:
        super().__init__(PolicyInfo(name="agent-like", action_space=Box(shape=(1,))))

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        self.info = PolicyInfo(
            name=self.info.name,
            action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,
            control_hz=embodiment_info.control_hz,
        )


def test_embodiment_is_conformant_for_guardrails_and_agent_policies() -> None:
    info = BimanualFrankaEmbodiment().info
    report = check_embodiment(info)
    assert report.ok is True
    assert report.issues == ()
    assert_embodiment_conformant(info)


def test_matching_sixteen_d_policy_has_zero_errors_and_zero_warnings() -> None:
    info = PolicyInfo(name="dual", action_space=action_box(), observation_space=observation_space())
    report = check_compatibility(_Policy(info), BimanualFrankaEmbodiment())  # type: ignore[arg-type]
    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []


def test_builtin_cubepick_reach_is_realizable() -> None:
    info = PolicyInfo(name="dual", action_space=action_box(), observation_space=observation_space())
    task = resolve("task", "cubepick-reach")
    report = check_compatibility(_Policy(info), BimanualFrankaEmbodiment(), task)  # type: ignore[arg-type]
    assert report.errors == []


def test_single_arm_openpi_policy_is_a_hard_dimension_error() -> None:
    report = check_compatibility(OpenpiPolicy(), BimanualFrankaEmbodiment())
    assert report.ok is False
    assert any(issue.code == "action_dim" for issue in report.errors)


def test_binding_policy_becomes_compatible_after_bind() -> None:
    embodiment = BimanualFrankaEmbodiment()
    policy = _BindingPolicy()
    before = check_compatibility(policy, embodiment)  # type: ignore[arg-type]
    assert any(issue.code == "action_dim" for issue in before.errors)
    policy.bind(embodiment.info)
    after = check_compatibility(policy, embodiment)  # type: ignore[arg-type]
    assert after.ok is True
    assert after.errors == after.warnings == []
    assert policy.info.action_space.dim == 16
