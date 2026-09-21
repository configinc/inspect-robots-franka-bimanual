"""Inspect Robots adapters for Franka arms and OpenPI DROID policy servers.

The package registers embodiments ``franka`` and ``franka_bimanual`` and policy
``openpi``. The single-arm pair exposes one shared absolute 8-D joint-position
contract; the bimanual embodiment exposes the same contract twice over, left
arm first, as 16-D. Every component remains inert at construction.
"""

from __future__ import annotations

from inspect_robots_franka.config import FrankaConfig, OpenpiConfig
from inspect_robots_franka.config_bimanual import BimanualFrankaConfig
from inspect_robots_franka.embodiment import FrankaEmbodiment
from inspect_robots_franka.embodiment_bimanual import BimanualFrankaEmbodiment
from inspect_robots_franka.operator import OperatorIO
from inspect_robots_franka.packing import DIM_LABELS, STATE_KEY, TOTAL_DIM
from inspect_robots_franka.packing_bimanual import DIM_LABELS as BIMANUAL_DIM_LABELS
from inspect_robots_franka.packing_bimanual import TOTAL_DIM as BIMANUAL_TOTAL_DIM
from inspect_robots_franka.policy import OpenpiPolicy
from inspect_robots_franka.preflight import build, run_preflight

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("inspect-robots-franka")
except PackageNotFoundError:  # pragma: no cover - only in a non-installed source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "BIMANUAL_DIM_LABELS",
    "BIMANUAL_TOTAL_DIM",
    "DIM_LABELS",
    "STATE_KEY",
    "TOTAL_DIM",
    "BimanualFrankaConfig",
    "BimanualFrankaEmbodiment",
    "FrankaConfig",
    "FrankaEmbodiment",
    "OpenpiConfig",
    "OpenpiPolicy",
    "OperatorIO",
    "__version__",
    "build",
    "run_preflight",
]
