from __future__ import annotations

from inspect_robots_franka.config_bimanual import (
    DEFAULT_JOINT_HIGH,
    DEFAULT_JOINT_LOW,
    BimanualFrankaConfig,
)
from inspect_robots_franka.embodiment_bimanual import _DOCS, BimanualFrankaEmbodiment
from inspect_robots_franka.packing_bimanual import DIM_LABELS


def test_docs_name_every_dimension_exactly_once_in_bullets() -> None:
    docs = BimanualFrankaEmbodiment().info.docs
    assert docs is not None
    bullets = [line for line in docs.splitlines() if line.startswith("- ")]
    assert len(bullets) == len(DIM_LABELS) == 16
    for label in DIM_LABELS:
        assert sum(line.startswith(f"- {label}:") for line in bullets) == 1


def test_docs_explain_left_first_packing_and_collision_risk() -> None:
    docs = " ".join((BimanualFrankaEmbodiment().info.docs or "").split())
    assert "first eight slots belong to the left arm" in docs
    assert "colliding with each other" in docs


def test_docs_do_not_leak_numeric_joint_limits() -> None:
    docs = BimanualFrankaEmbodiment().info.docs or ""
    revolute = (*DEFAULT_JOINT_LOW[:7], *DEFAULT_JOINT_LOW[8:15])
    revolute += (*DEFAULT_JOINT_HIGH[:7], *DEFAULT_JOINT_HIGH[8:15])
    for value in revolute:
        assert str(value) not in docs


def test_docs_extra_is_stripped_and_appended_once() -> None:
    embodiment = BimanualFrankaEmbodiment(
        BimanualFrankaConfig(docs_extra="  arms face each other 0.9 m apart\n")
    )
    assert embodiment.info.docs == _DOCS + "\n\narms face each other 0.9 m apart"
    assert BimanualFrankaEmbodiment(BimanualFrankaConfig(docs_extra=" \n ")).info.docs == _DOCS
