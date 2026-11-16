"""The STS3215 humanoid (18 DoF, ~825 mm, ~2.1 kg) — robot cfg for mjlab.

The model is GENERATED from the printable frame by ``humanoid/build_model.py``
(see that file for the conventions and the mass assumptions). Nothing here is
shared with the Microduck: different servos (Feetech STS3215, not Dynamixel
XL330), different joint set, different observation size.

Joint / action order (= MJCF order, = the runtime's order)::

    0-4   left  leg : hip_roll, hip_pitch, hip_yaw, knee, ankle
    5-9   right leg : hip_roll, hip_pitch, hip_yaw, knee, ankle
    10    torso_yaw
    11    neck_pitch
    12-14 left  arm : shoulder_pitch, shoulder_roll, elbow
    15-17 right arm : shoulder_pitch, shoulder_roll, elbow

Every joint angle is the servo's own reading (right-handed about the horn
axis, 0 = servo centre). Right-side servos are mirror images, so a mirrored
pose has the opposite sign on the right — e.g. right_knee = -left_knee.
"""

import os
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_microduck.actuator import Sts3215BamActuatorCfg

_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "humanoid"

HUMANOID_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
HUMANOID_SCENE_XML: Path = _ROBOT_DIR / "scene.xml"
assert HUMANOID_WALK_XML.exists(), f"XML not found: {HUMANOID_WALK_XML}"

HUMANOID_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_roll", "left_hip_pitch", "left_hip_yaw", "left_knee", "left_ankle",
    "right_hip_roll", "right_hip_pitch", "right_hip_yaw", "right_knee", "right_ankle",
    "torso_yaw", "neck_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
)
HUMANOID_NUM_JOINTS = len(HUMANOID_JOINT_NAMES)

# Bus IDs the runtime drives each joint with (Feetech STS3215, 1 Mbps).
# Tens digit = limb, unit = position along it. ID 1 (factory default) is unused
# on purpose, so a fresh servo can always be told apart.
HUMANOID_SERVO_IDS: tuple[int, ...] = (
    11, 12, 13, 14, 15,
    21, 22, 23, 24, 25,
    31, 32,
    41, 42, 43,
    51, 52, 53,
)

# Policy action -> joint target: target = HOME + ACTION_SCALE * action.
HUMANOID_ACTION_SCALE = 0.5

# Home: slightly crouched (knees 34°), feet flat, arms hanging 13° away from the
# body with a small elbow bend. Measured in sim (build_model + a 3 s hold): the
# pelvis sits at z = 0.453 m and the CoM ~8 mm behind the foot-sole centre, inside
# the support polygon. hip + knee + ankle pitch sum to 0 so the soles stay level.
HOME_JOINT_POS: dict[str, float] = {
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.30,
    "left_hip_yaw": 0.0,
    "left_knee": 0.60,
    "left_ankle": -0.30,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.30,
    "right_hip_yaw": 0.0,
    "right_knee": -0.60,
    "right_ankle": 0.30,
    "torso_yaw": 0.0,
    "neck_pitch": 0.0,
    "left_shoulder_pitch": 0.0,
    "left_shoulder_roll": -1.35,
    "left_elbow": 0.30,
    "right_shoulder_pitch": 0.0,
    "right_shoulder_roll": 1.35,
    "right_elbow": -0.30,
}
assert tuple(HOME_JOINT_POS) == HUMANOID_JOINT_NAMES

HUMANOID_STAND_Z = 0.453  # pelvis origin height at HOME, feet flat (measured)

HOME_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, HUMANOID_STAND_Z + 0.005),
    joint_pos={f"^{k}$": v for k, v in HOME_JOINT_POS.items()},
    joint_vel={".*": 0.0},
)

# Feet: box soles, condim 3, grippy. Every other part is an oriented box used for
# floor contact only: contype 2 / conaffinity 1 collides with the floor
# (contype 1) but not with the robot itself.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    contype=2,
    conaffinity=1,
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# BAM model of the 12 V STS3215. bam ships a fit of the 7.4 V servo only; this one is
# DERIVED from it by datasheet ratios and not yet fitted (see the json's "_derived"
# note) — refit with bam on the real servo. Firmware P = 32 is the factory default.
# 3S LiPo: 12.6 V full, ~11.1 V nominal, sagging under load.
STS3215_PARAMS_JSON = _ROBOT_DIR / "sts3215_12V_derived.json"
BATTERY_NOMINAL_V = 11.7
_BAM_ACTUATOR_KWARGS = dict(
    json_path=str(STS3215_PARAMS_JSON),
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=32.0,
    vin_range=(10.8, 12.6),
    vin_drop_gain_range=(0.0, 0.05),
    vin_min=10.0,
    delay_min_lag=3,
    delay_max_lag=6,
)


def load_sts3215_model():
    """The BAM model the policy trains against, for CPU tools (infer, the sim body)."""
    from bam.model import load_model

    return load_model(str(STS3215_PARAMS_JSON))


actuators = Sts3215BamActuatorCfg(**_BAM_ACTUATOR_KWARGS)


def get_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(HUMANOID_WALK_XML))


HUMANOID_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
