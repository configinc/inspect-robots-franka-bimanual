"""Validated configuration and shared spaces for a left and right Franka pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)

from inspect_robots_franka import config as arm
from inspect_robots_franka.config import FrankaConfig, _FromKwargs
from inspect_robots_franka.packing_bimanual import (
    ARM_DIM,
    ARMS,
    DIM_LABELS,
    STATE_KEY,
    TOTAL_DIM,
    tile,
)

DEFAULT_CAMERAS: tuple[str, ...] = ("exterior_cam", "left_wrist_cam", "right_wrist_cam")

DEFAULT_JOINT_LOW: tuple[float, ...] = tile(arm.DEFAULT_JOINT_LOW)
DEFAULT_JOINT_HIGH: tuple[float, ...] = tile(arm.DEFAULT_JOINT_HIGH)
DEFAULT_HOME_POSE: tuple[float, ...] = tile(arm.DEFAULT_HOME_POSE)

ACTION_SEMANTICS = ActionSemantics(
    control_mode="joint_pos",
    rotation_repr="none",
    gripper="continuous",
    frame="base",
    dim_labels=DIM_LABELS,
)

STATE_SPEC = StateSpec(
    fields=(StateField(key=STATE_KEY, shape=(TOTAL_DIM,), unit="rad+normalized"),)
)

Device = str | int | None


@dataclass(frozen=True)
class BimanualFrankaConfig(_FromKwargs):
    """Static hardware, safety, pacing, and camera configuration for two arms.

    Every 16-D field is the single-arm 8-D field repeated, left half first. The
    shared scalar fields (rate, dynamics, gripper, attendance, resolution) apply
    to both arms. ``arm_config`` slices one side into a ``FrankaConfig`` so the
    single-arm driver factory and validation rules serve both arms unchanged.
    """

    _FLOAT_TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"joint_low", "joint_high", "home_pose", "rest_pose"}
    )

    left_hostname: str | None = None
    right_hostname: str | None = None
    control_hz: float = 15.0
    joint_low: tuple[float, ...] = DEFAULT_JOINT_LOW
    joint_high: tuple[float, ...] = DEFAULT_JOINT_HIGH
    home_pose: tuple[float, ...] = DEFAULT_HOME_POSE
    rest_pose: tuple[float, ...] | None = None
    relative_dynamics_factor: float = 0.15
    gripper_max_width: float = 0.08
    gripper_speed: float = 0.05
    gripper_deadband: float = 0.1
    unattended: bool = False
    exterior_cam_device: str | int | None = None
    left_wrist_cam_device: str | int | None = None
    right_wrist_cam_device: str | int | None = None
    cam_height: int = 480
    cam_width: int = 640
    docs_extra: str = ""

    def __post_init__(self) -> None:
        """Reject configurations that violate the fixed 16-D safety contract.

        Lengths are checked here. Every per-arm rule (finite ordered bounds,
        poses inside the bounds, gripper and dynamics parameters, resolution)
        is delegated to ``FrankaConfig`` through ``arm_config`` so both
        embodiments enforce one rule set.
        """
        for name in ("joint_low", "joint_high", "home_pose"):
            if len(getattr(self, name)) != TOTAL_DIM:
                raise ValueError(f"{name} must have {TOTAL_DIM} entries")
        if self.rest_pose is not None and len(self.rest_pose) != TOTAL_DIM:
            raise ValueError(f"rest_pose must have {TOTAL_DIM} entries")
        for side in ARMS:
            try:
                self.arm_config(side)
            except ValueError as exc:
                raise ValueError(f"{side} arm: {exc}") from exc

    def arm_config(self, side: str) -> FrankaConfig:
        """Return the single-arm configuration slice that drives one side."""
        if side not in ARMS:
            raise ValueError(f"side must be one of {ARMS}, got {side!r}")
        start = ARMS.index(side) * ARM_DIM
        stop = start + ARM_DIM
        hostname: str | None = getattr(self, f"{side}_hostname")
        return FrankaConfig(
            hostname=hostname,
            control_hz=self.control_hz,
            joint_low=tuple(self.joint_low[start:stop]),
            joint_high=tuple(self.joint_high[start:stop]),
            home_pose=tuple(self.home_pose[start:stop]),
            rest_pose=None if self.rest_pose is None else tuple(self.rest_pose[start:stop]),
            relative_dynamics_factor=self.relative_dynamics_factor,
            gripper_max_width=self.gripper_max_width,
            gripper_speed=self.gripper_speed,
            gripper_deadband=self.gripper_deadband,
            unattended=self.unattended,
            cam_height=self.cam_height,
            cam_width=self.cam_width,
        )

    @property
    def camera_devices(self) -> dict[str, Device]:
        """Return the declared camera names mapped to their configured devices."""
        return {
            "exterior_cam": self.exterior_cam_device,
            "left_wrist_cam": self.left_wrist_cam_device,
            "right_wrist_cam": self.right_wrist_cam_device,
        }

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Return configured lower action bounds as float64."""
        return np.asarray(self.joint_low, dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Return configured upper action bounds as float64."""
        return np.asarray(self.joint_high, dtype=np.float64)


def camera_specs(height: int, width: int) -> tuple[CameraSpec, ...]:
    """Build the three camera declarations at a shared resolution."""
    return tuple(
        CameraSpec(name=name, height=height, width=width, channels=3) for name in DEFAULT_CAMERAS
    )


def action_box(cfg: BimanualFrankaConfig | None = None) -> Box:
    """Build the shared 16-D absolute joint-position space, optionally with rig bounds."""
    return Box(
        shape=(TOTAL_DIM,),
        low=cfg.low if cfg is not None else None,
        high=cfg.high if cfg is not None else None,
        semantics=ACTION_SEMANTICS,
    )


def observation_space(cfg: BimanualFrankaConfig | None = None) -> ObservationSpace:
    """Build the three-camera and packed-proprioception observation contract."""
    height = cfg.cam_height if cfg is not None else 480
    width = cfg.cam_width if cfg is not None else 640
    return ObservationSpace(cameras=camera_specs(height, width), state=STATE_SPEC)
