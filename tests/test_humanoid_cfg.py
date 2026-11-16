"""Invariants of the STS3215 humanoid model and its velocity task (CPU only)."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab_microduck.robot.humanoid_constants import (
    HOME_JOINT_POS,
    HUMANOID_JOINT_NAMES,
    HUMANOID_SCENE_XML,
    HUMANOID_SERVO_IDS,
    HUMANOID_STAND_Z,
    HUMANOID_WALK_XML,
    _BAM_ACTUATOR_KWARGS,
)
from mjlab_microduck.tasks.humanoid_velocity_env_cfg import make_humanoid_velocity_env_cfg

ROBOT_DIR = Path(HUMANOID_WALK_XML).parent


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(HUMANOID_SCENE_XML))


def test_joint_order_is_the_contract(model):
    """MJCF order = action order = the runtime's table. 18 hinges after the free joint."""
    names = tuple(model.joint(i).name for i in range(1, model.njnt))
    assert names == HUMANOID_JOINT_NAMES
    assert tuple(model.actuator(i).name for i in range(model.nu)) == HUMANOID_JOINT_NAMES
    assert len(HUMANOID_SERVO_IDS) == len(HUMANOID_JOINT_NAMES) == 18
    assert len(set(HUMANOID_SERVO_IDS)) == 18
    assert 1 not in HUMANOID_SERVO_IDS  # factory ID stays free for a replacement servo


def test_home_is_inside_the_joint_ranges(model):
    for name, q in HOME_JOINT_POS.items():
        lo, hi = model.joint(name).range
        assert lo < q < hi, (name, q, lo, hi)


def test_home_is_mirror_symmetric():
    """Mirrored servos: the right side reads the negated left angle."""
    for name, q in HOME_JOINT_POS.items():
        if name.startswith("left_"):
            assert HOME_JOINT_POS["right_" + name[5:]] == pytest.approx(-q), name


def test_right_ranges_mirror_the_left(model):
    for name in HUMANOID_JOINT_NAMES:
        if name.startswith("left_"):
            lo, hi = model.joint(name).range
            rlo, rhi = model.joint("right_" + name[5:]).range
            assert (rlo, rhi) == pytest.approx((-hi, -lo)), name


def test_home_feet_are_flat_and_level_with_the_floor(model):
    """hip + knee + ankle pitch sum to zero, and HUMANOID_STAND_Z puts the soles on z = 0."""
    for side in ("left", "right"):
        s = sum(HOME_JOINT_POS[f"{side}_{j}"] for j in ("hip_pitch", "knee", "ankle"))
        assert s == pytest.approx(0.0, abs=1e-9)
    d = mujoco.MjData(model)
    d.qpos[2] = HUMANOID_STAND_Z
    for n, q in HOME_JOINT_POS.items():
        d.joint(n).qpos = q
    mujoco.mj_kinematics(model, d)
    for side in ("left", "right"):
        assert d.site(f"{side}_foot").xpos[2] == pytest.approx(0.0, abs=2e-3)
        z = d.xmat[model.body(f"foot_{side[0].upper()}").id].reshape(3, 3)[:, 2]
        assert z[2] == pytest.approx(1.0, abs=1e-6)


def test_knee_flexes_backward_and_hip_pitch_lifts_the_leg_forward(model):
    """Positive left knee = flexion; negative left hip pitch = leg forward (README ranges)."""
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    ankle0 = d.body("foot_L").xpos.copy()
    d.joint("left_knee").qpos = 0.5
    mujoco.mj_kinematics(model, d)
    assert d.body("foot_L").xpos[0] < ankle0[0] - 0.05
    d.joint("left_knee").qpos = 0.0
    d.joint("left_hip_pitch").qpos = -0.5
    mujoco.mj_kinematics(model, d)
    assert d.body("foot_L").xpos[0] > ankle0[0] + 0.1


def test_mass_model_is_plausible(model):
    total = model.body_subtreemass[model.body("pelvis").id]
    assert 1.8 < total < 2.6
    report = json.loads((ROBOT_DIR / "mass_report.json").read_text())
    assert report["total_mass_kg"] == pytest.approx(total, abs=0.01)


def test_bam_actuator_is_the_12v_sts3215_on_3s():
    from bam.model import load_model

    assert _BAM_ACTUATOR_KWARGS["kp_fw"] == 32.0
    lo, hi = _BAM_ACTUATOR_KWARGS["vin_range"]
    assert 10.5 <= lo < hi <= 12.6  # a 3S pack: never above 12.6 V full
    m12 = load_model(_BAM_ACTUATOR_KWARGS["json_path"])
    m7 = load_model(motor_name="feetech_sts3215_7_4V", model="m1")
    assert m12.actuator_name == "sts3215"
    # Datasheet ratios the derivation claims: same no-load speed at rated volts, x30/19.5 stall.
    assert 12.0 / m12.kt.value == pytest.approx(7.4 / m7.kt.value, rel=1e-6)
    stall = lambda m, v: v * m.kt.value / m.R.value
    assert stall(m12, 12.0) / stall(m7, 7.4) == pytest.approx(30.0 / 19.5, rel=1e-6)


def test_env_cfg_wiring():
    cfg = make_humanoid_velocity_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events  # BAM needs per-world friction fields
    assert set(cfg.observations["actor"].terms) == {
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command",
    }
    assert list(cfg.observations["actor"].terms) == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command",
    ]
    assert cfg.rewards["upright"].params["asset_cfg"].body_names == ("chest",)
    # mjlab-base cost functions return >= 0 -> negative weights.
    for name in ("action_rate_l2", "dof_pos_limits", "foot_clearance", "foot_swing_height", "foot_slip",
                 "body_ang_vel", "angular_momentum"):
        assert cfg.rewards[name].weight < 0, name
    for name in ("track_linear_velocity", "track_angular_velocity", "upright", "pose", "air_time"):
        assert cfg.rewards[name].weight > 0, name
    assert cfg.commands["twist"].rel_standing_envs > 0  # zero command is trained
    assert "nan_state" in cfg.terminations and "body_on_ground" in cfg.terminations


def test_play_cfg_builds():
    cfg = make_humanoid_velocity_env_cfg(play=True)
    assert cfg.curriculum == {}
    assert not cfg.observations["actor"].enable_corruption
