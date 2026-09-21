from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_franka import packing, packing_bimanual


def test_constants_and_labels_repeat_the_single_arm_packing_left_first() -> None:
    assert packing_bimanual.ARMS == ("left", "right")
    assert packing_bimanual.NUM_ARMS == 2
    assert packing_bimanual.ARM_DIM == packing.TOTAL_DIM == 8
    assert packing_bimanual.TOTAL_DIM == 16
    assert packing_bimanual.STATE_KEY == packing.STATE_KEY == "joint_pos"
    assert packing_bimanual.DIM_LABELS[:8] == tuple(f"left_{label}" for label in packing.DIM_LABELS)
    assert packing_bimanual.DIM_LABELS[8:] == tuple(
        f"right_{label}" for label in packing.DIM_LABELS
    )
    assert packing_bimanual.DIM_LABELS[7] == "left_gripper"
    assert packing_bimanual.DIM_LABELS[15] == "right_gripper"
    assert len(set(packing_bimanual.DIM_LABELS)) == packing_bimanual.TOTAL_DIM


def test_validate_dim_accepts_list_and_returns_float64() -> None:
    out = packing_bimanual.validate_dim(list(range(16)))
    assert np.array_equal(out, np.arange(16))
    assert out.dtype == np.float64


@pytest.mark.parametrize("bad", [np.zeros(8), np.zeros(15), np.zeros(17), np.zeros((2, 8))])
def test_validate_dim_rejects_wrong_shape(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="expected a 16-D vector"):
        packing_bimanual.validate_dim(bad)


def test_split_arms_returns_independent_copies() -> None:
    vec = np.arange(16, dtype=float)
    left, right = packing_bimanual.split_arms(vec)
    assert np.array_equal(left, np.arange(8))
    assert np.array_equal(right, np.arange(8, 16))
    left[0] = 99.0
    right[0] = 99.0
    assert vec[0] == 0.0
    assert vec[8] == 8.0
    assert packing.gripper(left) == 7.0
    assert packing.gripper(right) == 15.0


def test_join_arms_packs_left_first_and_validates_each_half() -> None:
    joined = packing_bimanual.join_arms(np.arange(8), np.arange(8, 16))
    assert np.array_equal(joined, np.arange(16))
    assert joined.dtype == np.float64
    with pytest.raises(ValueError, match="expected an 8-D vector"):
        packing_bimanual.join_arms(np.zeros(16), np.zeros(8))


def test_tile_repeats_a_per_arm_tuple_and_rejects_wrong_length() -> None:
    tiled = packing_bimanual.tile(tuple(range(8)))
    assert tiled == (*range(8), *range(8))
    assert all(isinstance(value, float) for value in tiled)
    with pytest.raises(ValueError, match="expected an 8-D vector"):
        packing_bimanual.tile((0.0,) * 16)
