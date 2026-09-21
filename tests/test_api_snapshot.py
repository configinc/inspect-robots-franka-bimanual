from __future__ import annotations

import re

import inspect_robots_franka

EXPECTED_API = {
    "BimanualFrankaConfig",
    "BimanualFrankaEmbodiment",
    "BimanualRobotEnvEmbodiment",
    "BIMANUAL_DIM_LABELS",
    "BIMANUAL_TOTAL_DIM",
    "FrankaConfig",
    "OpenpiConfig",
    "FrankaEmbodiment",
    "OpenpiPolicy",
    "OperatorIO",
    "STATE_KEY",
    "TOTAL_DIM",
    "Y_FRAME_ROBOTIQ_HOME_POSE",
    "DIM_LABELS",
    "build",
    "run_preflight",
    "__version__",
}


def test_public_api_is_exact_and_importable() -> None:
    assert set(inspect_robots_franka.__all__) == EXPECTED_API
    for name in inspect_robots_franka.__all__:
        assert hasattr(inspect_robots_franka, name)


def test_version_is_tag_derived_shape() -> None:
    assert re.match(r"\d+\.\d+", inspect_robots_franka.__version__)


def test_entry_points_resolve() -> None:
    from inspect_robots.registry import resolve

    assert resolve("policy", "bimanual_hold").info.action_space.dim == 16
    assert resolve("policy", "openpi").info.name == "openpi"
    assert resolve("embodiment", "franka").info.name == "franka"
    assert resolve("embodiment", "franka_bimanual").info.name == "franka_bimanual"
    assert resolve("embodiment", "franka_bimanual").info.action_space.dim == 16
    robotenv = resolve("embodiment", "franka_bimanual_robotenv")
    assert robotenv.info.name == "franka_bimanual_robotenv"
    assert robotenv.info.action_space.dim == 16
