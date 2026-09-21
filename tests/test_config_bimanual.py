from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_franka import config as arm
from inspect_robots_franka.config_bimanual import (
    ACTION_SEMANTICS,
    DEFAULT_CAMERAS,
    DEFAULT_HOME_POSE,
    DEFAULT_JOINT_HIGH,
    DEFAULT_JOINT_LOW,
    BimanualFrankaConfig,
    action_box,
    camera_specs,
    observation_space,
)
from inspect_robots_franka.packing_bimanual import DIM_LABELS, STATE_KEY, TOTAL_DIM


def test_defaults_tile_the_single_arm_defaults_left_first() -> None:
    cfg = BimanualFrankaConfig()
    assert cfg.control_hz == 15.0
    assert cfg.joint_low == DEFAULT_JOINT_LOW == arm.DEFAULT_JOINT_LOW * 2
    assert cfg.joint_high == DEFAULT_JOINT_HIGH == arm.DEFAULT_JOINT_HIGH * 2
    assert cfg.home_pose == DEFAULT_HOME_POSE == arm.DEFAULT_HOME_POSE * 2
    assert cfg.low.shape == cfg.high.shape == (TOTAL_DIM,)
    assert np.all(cfg.low < cfg.high)
    assert (cfg.left_hostname, cfg.right_hostname) == (None, None)
    assert cfg.rest_pose is None
    assert cfg.camera_devices == {
        "exterior_cam": None,
        "left_wrist_cam": None,
        "right_wrist_cam": None,
    }


def test_arm_config_slices_each_side_and_carries_shared_fields() -> None:
    cfg = BimanualFrankaConfig(
        left_hostname="left.local",
        right_hostname="right.local",
        control_hz=10.0,
        gripper_deadband=0.2,
        unattended=True,
        cam_height=12,
        cam_width=16,
        rest_pose=DEFAULT_HOME_POSE,
        exterior_cam_device="/dev/exterior",
    )
    left = cfg.arm_config("left")
    right = cfg.arm_config("right")
    assert (left.hostname, right.hostname) == ("left.local", "right.local")
    assert left.joint_low == right.joint_low == arm.DEFAULT_JOINT_LOW
    assert left.joint_high == right.joint_high == arm.DEFAULT_JOINT_HIGH
    assert left.home_pose == right.home_pose == arm.DEFAULT_HOME_POSE
    assert left.rest_pose == right.rest_pose == arm.DEFAULT_HOME_POSE
    assert (left.control_hz, left.gripper_deadband, left.unattended) == (10.0, 0.2, True)
    assert (left.cam_height, left.cam_width) == (12, 16)
    assert left.exterior_cam_device is None
    assert left.wrist_cam_device is None
    assert BimanualFrankaConfig().arm_config("right").rest_pose is None
    with pytest.raises(ValueError, match="side must be one of"):
        cfg.arm_config("middle")


def test_asymmetric_bounds_stay_on_their_own_side() -> None:
    low = list(DEFAULT_JOINT_LOW)
    low[8] = -1.0
    cfg = BimanualFrankaConfig(joint_low=tuple(low))
    assert cfg.arm_config("right").joint_low[0] == -1.0
    assert cfg.arm_config("left").joint_low[0] == arm.DEFAULT_JOINT_LOW[0]
    assert cfg.low[8] == -1.0


def test_from_kwargs_parses_strings_and_rejects_single_arm_keys() -> None:
    home = ",".join(str(value) for value in DEFAULT_HOME_POSE)
    cfg = BimanualFrankaConfig.from_kwargs(
        left_hostname="172.16.0.2",
        right_hostname="172.16.1.2",
        home_pose=home,
        rest_pose=home,
        control_hz="20.0",
        unattended="true",
        cam_height="720",
    )
    assert cfg.home_pose == pytest.approx(DEFAULT_HOME_POSE)
    assert cfg.rest_pose == pytest.approx(DEFAULT_HOME_POSE)
    assert (cfg.left_hostname, cfg.right_hostname) == ("172.16.0.2", "172.16.1.2")
    assert cfg.control_hz == 20.0
    assert cfg.unattended is True
    assert cfg.cam_height == 720
    with pytest.raises(TypeError, match="unexpected config keys"):
        BimanualFrankaConfig.from_kwargs(hostname="172.16.0.2")
    with pytest.raises(TypeError, match="unexpected config keys"):
        BimanualFrankaConfig.from_kwargs(wrist_cam_device="/dev/video1")
    with pytest.raises(ValueError, match="home_pose must be a comma-separated"):
        BimanualFrankaConfig.from_kwargs(home_pose="0,bad")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"control_hz": 0.0}, "left arm: control_hz"),
        ({"joint_low": (0.0,) * 8}, "joint_low must have 16"),
        ({"joint_high": (0.0,) * 8}, "joint_high must have 16"),
        ({"home_pose": (0.0,) * 8}, "home_pose must have 16"),
        ({"rest_pose": (0.0,) * 8}, "rest_pose must have 16"),
        ({"joint_low": (*DEFAULT_JOINT_LOW[:-1], np.nan)}, "right arm: joint_low and joint_high"),
        ({"joint_low": DEFAULT_JOINT_HIGH}, "left arm: joint_low must be below"),
        ({"home_pose": (*DEFAULT_HOME_POSE[:-1], 2.0)}, "right arm: home_pose must be finite"),
        ({"rest_pose": (*DEFAULT_HOME_POSE[:-1], -1.0)}, "right arm: rest_pose must be finite"),
        ({"gripper_max_width": 0.0}, "left arm: gripper_max_width"),
        ({"gripper_speed": np.inf}, "left arm: gripper_speed"),
        ({"relative_dynamics_factor": 1.1}, "left arm: relative_dynamics_factor"),
        ({"gripper_deadband": 1.0}, "left arm: gripper_deadband"),
        ({"cam_height": 0}, "left arm: cam_height"),
    ],
)
def test_validation_names_the_offending_arm(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BimanualFrankaConfig(**kwargs)


def test_shared_spaces_have_pinned_semantics_state_and_three_cameras() -> None:
    cfg = BimanualFrankaConfig(cam_height=12, cam_width=16)
    box = action_box(cfg)
    assert box.shape == (16,)
    assert np.array_equal(box.low, cfg.low)
    assert np.array_equal(box.high, cfg.high)
    assert box.semantics is ACTION_SEMANTICS
    assert ACTION_SEMANTICS.control_mode == "joint_pos"
    assert ACTION_SEMANTICS.dim_labels == DIM_LABELS
    assert action_box().low is action_box().high is None
    obs = observation_space(cfg)
    assert obs.camera_names == frozenset(DEFAULT_CAMERAS)
    assert obs.state_keys == frozenset({STATE_KEY})
    assert obs.state is not None
    assert obs.state.fields[0].shape == (16,)
    assert obs.state.fields[0].unit == "rad+normalized"
    specs = camera_specs(12, 16)
    assert [(item.name, item.height, item.width) for item in specs] == [
        ("exterior_cam", 12, 16),
        ("left_wrist_cam", 12, 16),
        ("right_wrist_cam", 12, 16),
    ]
    assert observation_space().cameras[0].height == 480
