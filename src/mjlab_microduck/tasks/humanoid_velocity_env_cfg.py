"""Velocity (walking) task for the STS3215 humanoid.

A separate policy family from the Microduck: 18 joints instead of 14, so the
observation is 63-D and the action 18-D, and none of the duck's head/body-pose
command slots exist. Actor observation layout (the runtime builds exactly this)::

    [ base_ang_vel(3), projected_gravity(3), joint_pos - HOME (18),
      joint_vel (18), last_action (18), twist_command (3) ]          = 63

The recipe follows the Microduck velocity task where the lesson is about the
sim2real method (BAM actuators, non-accumulating DR, IMU misalignment, encoder
bias, sensor lags, NaN guards, zero-command and turn-in-place buckets) and the
mjlab G1 task where it is about being a humanoid (arm/waist posture stds, foot
clearance). Robot-specific numbers (speeds, foot heights, stds) are first
guesses scaled to an 825 mm, 2.1 kg robot — expect to tune them on the first
real run.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.humanoid_constants import (
    HUMANOID_ACTION_SCALE,
    HUMANOID_STAND_Z,
    HUMANOID_WALK_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp

NUM_STEPS_PER_ENV = 24

# Command ranges (m/s, rad/s). The robot's leg is ~0.47 m; 0.5 m/s forward is a
# brisk walk for it. Fixed ranges, like the duck (a widening curriculum outpaced it).
LIN_VEL_X = (-0.3, 0.5)
LIN_VEL_Y = (-0.2, 0.2)
ANG_VEL_Z = (-1.0, 1.0)
TURN_IN_PLACE_FRACTION = 0.15

FOOT_CLEARANCE_M = 0.04

# Domain randomization
ENABLE_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

COM_RANDOMIZATION_RANGE = 0.005          # ramped to ±15 mm by curriculum (chest)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.9, 1.1)  # the mass model is an estimate: ±10 %
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.8, 1.2)  # one fitted BAM model for every servo
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 4.0
ENCODER_BIAS_RANGE = (-0.015, 0.015)     # STS3215: 4096 counts/rev, ±0.86° mounting/zero error

# Bodies carrying the big unknown masses: battery + computer (chest), LiDAR (head).
HEAVY_BODIES = ("chest", "head", "pelvis")


def make_humanoid_velocity_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    site_names = ("left_foot", "right_foot")
    foot_geoms = ("left_foot_collision", "right_foot_collision")

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Anything but a foot touching the floor: used as a termination (a fall) —
    # the robot has no protective-fall skill, so ground contact with the knees,
    # hands or torso ends the episode instead of being farmed.
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(?!left_foot_collision$|right_foot_collision$).*_collision$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="maxforce",
        num_slots=1,
    )
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.05, num_samples=4),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": HUMANOID_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, body_ground_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "chest"

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = HUMANOID_ACTION_SCALE

    # ── rewards ─────────────────────────────────────────────────────────────
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("chest",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    cfg.rewards["pose"].params["std_standing"] = {
        r".*hip_roll.*": 0.05, r".*hip_yaw.*": 0.08, r".*hip_pitch.*": 0.1,
        r".*knee.*": 0.1, r".*ankle.*": 0.08,
        r"^torso_yaw$": 0.08, r"^neck_pitch$": 0.05,
        r".*shoulder.*": 0.1, r".*elbow.*": 0.15,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_roll.*": 0.1, r".*hip_yaw.*": 0.15, r".*hip_pitch.*": 0.35,
        r".*knee.*": 0.4, r".*ankle.*": 0.25,
        r"^torso_yaw$": 0.2, r"^neck_pitch$": 0.08,
        r".*shoulder_pitch.*": 0.3, r".*shoulder_roll.*": 0.15, r".*elbow.*": 0.3,
    }
    cfg.rewards["pose"].params["std_running"] = cfg.rewards["pose"].params["std_walking"]
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("chest",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = site_names
    for name in ("foot_clearance", "foot_swing_height", "foot_slip"):
        cfg.rewards[name].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = FOOT_CLEARANCE_M
    cfg.rewards["foot_swing_height"].params["target_height"] = FOOT_CLEARANCE_M
    cfg.rewards["foot_slip"].weight = -0.1

    cfg.rewards["air_time"].weight = 1.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.15
    cfg.rewards["air_time"].params["threshold_max"] = 0.40

    cfg.rewards["action_rate_l2"].weight = -0.05  # ramped by curriculum after the gait exists
    cfg.rewards.pop("soft_landing", None)

    # ── events ──────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
    cfg.events["foot_friction"].params["ranges"] = (0.5, 1.2)
    cfg.events.pop("base_com", None)  # replaced by the reset-time chest CoM DR below

    cfg.events["reset_base"].params["pose_range"]["z"] = (HUMANOID_STAND_Z + 0.002, HUMANOID_STAND_Z + 0.01)
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 2.0) if play else VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    else:
        cfg.events.pop("push_robot", None)

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAVY_BODIES),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        lo, hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAVY_BODIES),
                "alpha_range": (math.log(lo) / 2.0, math.log(hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE},
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # ── observations ────────────────────────────────────────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel)

    actor = cfg.observations["actor"].terms
    for k in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
        actor[k] = deepcopy(actor[k])
    for k in ("base_ang_vel", "projected_gravity"):
        actor[k].delay_min_lag = 0
        actor[k].delay_max_lag = 1
        actor[k].delay_update_period = 64
    # Feetech present-speed is a firmware estimate one bus read old.
    actor["joint_vel"].delay_min_lag = 1
    actor["joint_vel"].delay_max_lag = 1
    actor["joint_vel"].delay_update_period = 0

    actor["base_ang_vel"].noise = Unoise(n_min=-0.05, n_max=0.05)
    actor["projected_gravity"].noise = Unoise(n_min=-0.02, n_max=0.02)
    actor["joint_pos"].noise = Unoise(n_min=-0.003, n_max=0.003)  # 1.5 counts
    actor["joint_vel"].noise = Unoise(n_min=-0.3, n_max=0.3)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        actor["base_ang_vel"].func = microduck_mdp.base_ang_vel_imu_misaligned
        actor["base_ang_vel"].params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        actor["projected_gravity"].func = microduck_mdp.projected_gravity_imu_misaligned
        actor["projected_gravity"].params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            t = deepcopy(cfg.observations[grp].terms[term])
            t.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
            cfg.observations[grp].terms[term] = t

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for term, safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[term].func = safe

    # ── commands ────────────────────────────────────────────────────────────
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    command.rel_standing_envs = 0.1   # zero command = the deployment idle state; train it from step 0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = LIN_VEL_X
    command.ranges.lin_vel_y = LIN_VEL_Y
    command.ranges.ang_vel_z = ANG_VEL_Z
    command.viz.z_offset = 0.9
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    cfg.commands["twist"].rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION

    # ── terminations ────────────────────────────────────────────────────────
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)
    cfg.terminations["body_on_ground"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": body_ground_cfg.name},
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    # ── terrain ─────────────────────────────────────────────────────────────
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum.pop("terrain_levels", None)
    else:
        from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
            MICRODUCK_ROUGH_TERRAINS_CFG,
            _soften_terrain_contacts,
        )

        # The duck's gentle terrain (≤1.5 cm steps) doubled: this robot is 3× taller.
        terrain = deepcopy(MICRODUCK_ROUGH_TERRAINS_CFG)
        terrain.sub_terrains["pyramid_stairs"].step_height_range = (0.0, 0.03)
        terrain.sub_terrains["random_grid"].grid_height_range = (0.0, 0.02)
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = terrain
        cfg.scene.spec_fn = _soften_terrain_contacts
        cfg.sim.nconmax = 200
        cfg.sim.mujoco.iterations = 30
        cfg.sim.mujoco.ls_iterations = 50
        if play:
            terrain.curriculum = False
            terrain.num_cols = 5
            terrain.num_rows = 5

    # ── curricula ───────────────────────────────────────────────────────────
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.05},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.1},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.4},
            ],
        },
    )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": COM_RANDOMIZATION_RANGE},
                    {"step": 750 * NUM_STEPS_PER_ENV, "range": 0.01},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "range": 0.015},
                ],
            },
        )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

    return cfg


HumanoidRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_humanoid",
    experiment_name="humanoid_velocity",
    run_name="humanoid_velocity",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=6_000,
)
