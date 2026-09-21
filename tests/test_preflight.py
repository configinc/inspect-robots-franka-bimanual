from __future__ import annotations

import json
from typing import Any

import pytest
from inspect_robots.compat import CompatibilityReport, CompatIssue
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import policy as register_policy
from inspect_robots.spaces import Box

from inspect_robots_franka import preflight


def _report(severity: str | None = None) -> CompatibilityReport:
    issues = [] if severity is None else [CompatIssue(severity, "code", "detail")]
    return CompatibilityReport(issues=issues)


@register_policy("preflight-test-binding-policy")
class _BindingPolicy:
    """Adopt the embodiment contract at bind time, as LLM agent policies do."""

    config = PolicyConfig()

    def __init__(self, **kwargs: str) -> None:
        self.info = PolicyInfo(name="binding", action_space=Box(shape=(1,)))
        self.bound_to: str | None = None
        self.kwargs = kwargs

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        self.bound_to = embodiment_info.name
        self.info = PolicyInfo(
            name="binding",
            action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,
            control_hz=embodiment_info.control_hz,
        )

    def reset(self, scene: object) -> None:
        return None

    def act(self, observation: object) -> Any:
        raise AssertionError("not called")


def test_build_and_default_preflight_are_inert_and_compatible() -> None:
    policy, embodiment = preflight.build()
    assert policy.info.name == "openpi"
    assert embodiment.info.name == "franka"
    assert preflight.build_bimanual().info.name == "franka_bimanual"
    report = preflight.run_preflight()
    assert report.errors == report.warnings == []


def test_build_embodiment_by_name_rejects_unknown_names() -> None:
    assert preflight.build_embodiment().info.name == "franka"
    assert preflight.build_embodiment("franka_bimanual").info.action_space.dim == 16
    with pytest.raises(ValueError, match="embodiment must be one of"):
        preflight.build_embodiment("yam_arms")


def test_openpi_against_bimanual_is_an_honest_dimension_error() -> None:
    report = preflight.run_preflight(embodiment_name="franka_bimanual")
    assert report.ok is False
    codes = [issue.code for issue in report.errors]
    assert "action_dim" in codes
    assert "missing_camera" in codes


def test_registry_policy_is_bound_before_the_check() -> None:
    embodiment = preflight.build_bimanual()
    policy = preflight.build_policy("preflight-test-binding-policy", embodiment.info)
    assert isinstance(policy, _BindingPolicy)
    assert policy.bound_to == "franka_bimanual"
    assert policy.info.action_space.dim == 16
    report = preflight.run_preflight(
        policy_name="preflight-test-binding-policy", embodiment_name="franka_bimanual"
    )
    assert report.ok is True
    assert report.errors == report.warnings == []
    assert preflight.build_policy("openpi", preflight.build()[1].info).info.name == "openpi"


def test_policy_args_reach_the_constructor_for_registry_and_openpi_policies() -> None:
    info = preflight.build_bimanual().info
    policy = preflight.build_policy(
        "preflight-test-binding-policy",
        info,
        {"model": "openai/gpt-6-astra", "effort": "low"},
    )
    assert policy.kwargs == {"model": "openai/gpt-6-astra", "effort": "low"}
    openpi = preflight.build_policy("openpi", info, {"name": "custom_openpi", "port": "9000"})
    assert openpi.info.name == "custom_openpi"
    report = preflight.run_preflight(
        policy_name="preflight-test-binding-policy",
        embodiment_name="franka_bimanual",
        policy_args={"model": "anthropic/claude-fable-5-1"},
    )
    assert report.ok is True


def test_parse_policy_args_accepts_values_with_equals_and_rejects_malformed() -> None:
    assert preflight.parse_policy_args([]) == {}
    assert preflight.parse_policy_args(["model=openai/gpt-6-astra", "base_url=http://x?a=b"]) == {
        "model": "openai/gpt-6-astra",
        "base_url": "http://x?a=b",
    }
    for bad in ("model", "=value"):
        with pytest.raises(ValueError, match="KEY=VALUE"):
            preflight.parse_policy_args([bad])


def test_preflight_resolves_optional_task_and_uses_injected_components() -> None:
    policy, embodiment = preflight.build()
    report = preflight.run_preflight("cubepick-reach", policy=policy, embodiment=embodiment)
    assert report.ok is True
    sentinel = _report("warning")
    assert preflight.run_preflight(check=lambda *_args, **_kwargs: sentinel) is sentinel


def test_injected_embodiment_skips_policy_resolution_for_that_name() -> None:
    embodiment = preflight.build_bimanual()
    report = preflight.run_preflight(
        embodiment=embodiment, policy_name="preflight-test-binding-policy"
    )
    assert report.ok is True


@pytest.mark.parametrize(
    ("report", "args", "code", "text"),
    [
        (_report(), [], 0, "OK: policy 'openpi' and embodiment 'franka' are compatible."),
        (_report("warning"), [], 0, "WARNING"),
        (_report("error"), [], 1, "INCOMPATIBLE: policy 'openpi' and embodiment 'franka'"),
        (_report(), ["--dry-run"], 0, "dry-run"),
        (
            _report(),
            ["--embodiment", "franka_bimanual", "--policy", "agent"],
            0,
            "policy 'agent' and embodiment 'franka_bimanual'",
        ),
    ],
)
def test_human_cli(
    report: CompatibilityReport,
    args: list[str],
    code: int,
    text: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert preflight.main(args, run=lambda *_args, **_kwargs: report) == code
    assert text in capsys.readouterr().out


def test_cli_forwards_names_to_the_runner() -> None:
    seen: dict[str, object] = {}

    def run(task: str | None, **kwargs: object) -> CompatibilityReport:
        seen["task"] = task
        seen.update(kwargs)
        return _report()

    argv = [
        "--embodiment",
        "franka_bimanual",
        "--policy",
        "agent",
        "-P",
        "model=openai/gpt-6-astra",
        "--policy-arg",
        "effort=low",
    ]
    assert preflight.main(argv, run=run) == 0
    assert seen == {
        "task": None,
        "policy_name": "agent",
        "embodiment_name": "franka_bimanual",
        "policy_args": {"model": "openai/gpt-6-astra", "effort": "low"},
    }


def test_cli_rejects_unknown_embodiment_and_malformed_policy_args(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        preflight.main(["--embodiment", "yam_arms"], run=lambda *_a, **_k: _report())
    with pytest.raises(SystemExit):
        preflight.main(["-P", "model"], run=lambda *_a, **_k: _report())
    assert "KEY=VALUE" in capsys.readouterr().err


def test_json_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert preflight.main(["--json"], run=lambda *_args, **_kwargs: _report("error")) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": False,
        "policy": "openpi",
        "embodiment": "franka",
        "errors": [{"code": "code", "message": "detail"}],
        "warnings": [],
    }


def test_main_default_run_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    assert preflight.main([]) == 0
    assert "OK:" in capsys.readouterr().out
