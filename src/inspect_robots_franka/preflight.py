"""Hardware-free compatibility preflight for the Franka embodiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from typing import Any

from inspect_robots.compat import CompatibilityReport, check_compatibility
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.registry import resolve
from inspect_robots.task import Task

from inspect_robots_franka.config import FrankaConfig, OpenpiConfig
from inspect_robots_franka.config_bimanual import BimanualFrankaConfig
from inspect_robots_franka.embodiment import FrankaEmbodiment
from inspect_robots_franka.embodiment_bimanual import BimanualFrankaEmbodiment
from inspect_robots_franka.policy import OpenpiPolicy

CheckFn = Callable[..., CompatibilityReport]
Embodiment = FrankaEmbodiment | BimanualFrankaEmbodiment

EMBODIMENTS: tuple[str, ...] = ("franka", "franka_bimanual")
DEFAULT_POLICY = "openpi"


def build(
    franka_cfg: FrankaConfig | None = None,
    openpi_cfg: OpenpiConfig | None = None,
) -> tuple[OpenpiPolicy, FrankaEmbodiment]:
    """Construct the single-arm policy and embodiment without hardware or network access."""
    return OpenpiPolicy(openpi_cfg), FrankaEmbodiment(franka_cfg)


def build_bimanual(cfg: BimanualFrankaConfig | None = None) -> BimanualFrankaEmbodiment:
    """Construct the two-arm embodiment without hardware or network access."""
    return BimanualFrankaEmbodiment(cfg)


def build_embodiment(name: str = "franka") -> Embodiment:
    """Construct one of this package's embodiments by registry name."""
    if name == "franka":
        return FrankaEmbodiment()
    if name == "franka_bimanual":
        return build_bimanual()
    raise ValueError(f"embodiment must be one of {EMBODIMENTS}, got {name!r}")


def build_policy(
    policy_name: str,
    embodiment_info: EmbodimentInfo,
    policy_args: Mapping[str, Any] | None = None,
) -> Any:
    """Resolve a registered policy and bind it to the embodiment's declared spaces.

    ``openpi`` is constructed directly. Any other name, for example ``agent``,
    resolves through the Inspect Robots registry. ``policy_args`` (the CLI's
    ``-P key=value`` pairs) go to the constructor either way. Policies that adopt
    the embodiment contract at bind time (LLM agent policies) expose ``bind``; it
    is called here so the report judges the spaces the rollout will actually use.
    """
    kwargs = dict(policy_args or {})
    if policy_name == DEFAULT_POLICY:
        policy: Any = OpenpiPolicy(**kwargs)
    else:
        policy = resolve("policy", policy_name, **kwargs)
    bind = getattr(policy, "bind", None)
    if callable(bind):
        bind(embodiment_info)
    return policy


def parse_policy_args(pairs: list[str]) -> dict[str, str]:
    """Turn repeated ``KEY=VALUE`` strings into constructor keyword arguments."""
    parsed: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"policy argument must look like KEY=VALUE, got {pair!r}")
        parsed[key] = value
    return parsed


def run_preflight(
    task_name: str | None = None,
    *,
    policy: Any | None = None,
    embodiment: Embodiment | None = None,
    check: CheckFn = check_compatibility,
    policy_name: str = DEFAULT_POLICY,
    embodiment_name: str = "franka",
    policy_args: dict[str, str] | None = None,
) -> CompatibilityReport:
    """Return compatibility findings, optionally including task realizability."""
    emb = embodiment if embodiment is not None else build_embodiment(embodiment_name)
    pol = policy if policy is not None else build_policy(policy_name, emb.info, policy_args)
    task: Task | None = resolve("task", task_name) if task_name else None
    return check(pol, emb, task)


def _format_human(
    report: CompatibilityReport, *, dry_run: bool, policy_name: str, embodiment_name: str
) -> str:
    pair = f"policy {policy_name!r} and embodiment {embodiment_name!r}"
    lines = [f"OK: {pair} are compatible." if report.ok else f"INCOMPATIBLE: {pair}"]
    for issue in report.errors:
        lines.append(f"  ERROR   [{issue.code}] {issue.message}")
    for issue in report.warnings:
        lines.append(f"  WARNING [{issue.code}] {issue.message}")
    if dry_run:
        lines.append("(dry-run) No motion will be commanded.")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, run: CheckFn | None = None) -> int:
    """Print a compatibility report and return nonzero only for errors."""
    parser = argparse.ArgumentParser(prog="inspect-robots-franka-preflight")
    parser.add_argument(
        "--task", default=None, help="optional task name to check scene realizability"
    )
    parser.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        help="registered policy name to check (default: openpi; use agent for an LLM policy)",
    )
    parser.add_argument(
        "--embodiment",
        default="franka",
        choices=EMBODIMENTS,
        help="which Franka embodiment to check (default: franka)",
    )
    parser.add_argument(
        "-P",
        "--policy-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="policy constructor argument, repeatable (e.g. -P model=openai/gpt-6-astra)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--dry-run", action="store_true", help="affirm no motion is commanded")
    args = parser.parse_args(argv)
    try:
        policy_args = parse_policy_args(args.policy_arg)
    except ValueError as exc:
        parser.error(str(exc))
    run_fn: Callable[..., CompatibilityReport] = run if run is not None else run_preflight
    report = run_fn(
        args.task,
        policy_name=args.policy,
        embodiment_name=args.embodiment,
        policy_args=policy_args,
    )
    if args.json:
        payload = {
            "ok": report.ok,
            "policy": args.policy,
            "embodiment": args.embodiment,
            "errors": [{"code": item.code, "message": item.message} for item in report.errors],
            "warnings": [{"code": item.code, "message": item.message} for item in report.warnings],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(
            _format_human(
                report,
                dry_run=args.dry_run,
                policy_name=args.policy,
                embodiment_name=args.embodiment,
            )
        )
    return 1 if report.errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
