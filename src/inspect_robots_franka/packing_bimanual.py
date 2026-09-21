"""Canonical 16-D joint-position packing for a left and right Franka pair.

The shared vector is ``[left_joint1, ..., left_joint7, left_gripper, right_joint1,
..., right_joint7, right_gripper]``: the single-arm 8-D packing repeated once per
arm, left first. Each half keeps the single-arm units, absolute radians for the
revolute slots and a normalized gripper with 0 closed and 1 open. This module is
pure NumPy so importing the package never loads a hardware stack.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from inspect_robots_franka import packing as arm

ARMS: tuple[str, ...] = ("left", "right")
NUM_ARMS = len(ARMS)
ARM_DIM = arm.TOTAL_DIM
TOTAL_DIM = NUM_ARMS * ARM_DIM
DIM_LABELS: tuple[str, ...] = tuple(f"{side}_{label}" for side in ARMS for label in arm.DIM_LABELS)
STATE_KEY = arm.STATE_KEY

Vec = npt.NDArray[np.float64]


def validate_dim(vec: npt.ArrayLike) -> Vec:
    """Return a one-dimensional float vector of length sixteen.

    Two-dimensional inputs are rejected instead of flattened because flattening
    can silently scramble a robot action's declared packing.
    """
    arr: Vec = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != TOTAL_DIM:
        raise ValueError(f"expected a {TOTAL_DIM}-D vector, got shape {np.shape(vec)}")
    return arr


def split_arms(vec: npt.ArrayLike) -> tuple[Vec, Vec]:
    """Return copies of the left and right 8-D halves, in ``ARMS`` order."""
    arr = validate_dim(vec)
    left: Vec = arr[:ARM_DIM].copy()
    right: Vec = arr[ARM_DIM:].copy()
    return left, right


def join_arms(left: npt.ArrayLike, right: npt.ArrayLike) -> Vec:
    """Pack two validated 8-D arm vectors into one 16-D vector, left first."""
    out: Vec = np.concatenate((arm.validate_dim(left), arm.validate_dim(right)))
    return out


def tile(per_arm: Sequence[float]) -> tuple[float, ...]:
    """Repeat one 8-D per-arm tuple for every arm, for shared defaults."""
    values = tuple(float(v) for v in arm.validate_dim(per_arm))
    return values * NUM_ARMS
