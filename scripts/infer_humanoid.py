"""Run an exported humanoid ONNX policy in CPU MuJoCo — the deployment rehearsal.

    uv run scripts/infer_humanoid.py policy.onnx --vx 0.3 --seconds 10
    uv run scripts/infer_humanoid.py policy.onnx --wz 0.8 --video walk.gif   # needs MUJOCO_GL=egl|osmesa
    uv run scripts/infer_humanoid.py --zero                                 # no policy: hold HOME

This is the robot's control loop with the servos simulated, written the way the
runtime (`microduck`'s `humanoid` profile) does it, so a mismatch shows up here
before it shows up on hardware:

  * 50 Hz policy, 200 Hz physics (the training env's timestep 0.005 × decimation 4);
  * the 63-D observation built from what the robot can measure — pelvis IMU gyro
    and gravity, joint positions relative to HOME, joint velocities, the last
    action, the twist command — in the order baked into the ONNX metadata;
  * targets = HOME + action_scale × action, read from the ONNX metadata too;
  * the STS3215 BAM actuator model on CPU (`--no-bam` falls back to the XML PD).

It prints how far the robot went against what was commanded and whether it fell.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.robot.humanoid_constants import (
    HOME_JOINT_POS,
    HUMANOID_ACTION_SCALE,
    HUMANOID_JOINT_NAMES,
    HUMANOID_SCENE_XML,
    HUMANOID_STAND_Z,
    _BAM_ACTUATOR_KWARGS,
)

PHYSICS_DT = 0.005
DECIMATION = 4
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


def load_policy(path: str | None):
    """(run(obs) -> action, metadata dict). ``None`` = the zero policy."""
    if path is None:
        meta = dict(
            joint_names=",".join(HUMANOID_JOINT_NAMES),
            default_joint_pos=",".join(str(HOME_JOINT_POS[n]) for n in HUMANOID_JOINT_NAMES),
            action_scale=str(HUMANOID_ACTION_SCALE),
            observation_names="base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command",
        )
        return (lambda obs: np.zeros(len(HUMANOID_JOINT_NAMES), np.float32)), meta
    import onnx
    import onnxruntime as ort

    meta = {p.key: p.value for p in onnx.load(path).metadata_props}
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return (lambda obs: sess.run(None, {name: obs[None].astype(np.float32)})[0][0]), meta


def build(no_bam: bool):
    spec = mujoco.MjSpec.from_file(str(HUMANOID_SCENE_XML))
    bam_model = None
    if not no_bam:
        from bam.model import load_model

        kw = _BAM_ACTUATOR_KWARGS
        bam_model = load_model(motor_name=kw["motor_name"], model=kw["model"])
        bam_model.actuator.kp = kw["kp_fw"]
        bam_model.actuator.vin = 7.8  # mid-charge 2S
        limit = bam_model.actuator.vin * bam_model.kt.value / bam_model.R.value
        for act in spec.actuators:
            act.set_to_motor()
            act.forcelimited = True
            act.forcerange = (-limit, limit)
            act.ctrllimited = False
            act.gear = [1.0, 0, 0, 0, 0, 0]
        for j in spec.joints:
            if j.name in HOME_JOINT_POS:
                j.damping = np.zeros((3, 1))
                j.frictionloss = 0.0
                j.solref_friction = STIFF_SOLREF_FRICTION
                j.solimp_friction = STIFF_SOLIMP_FRICTION
    model = spec.compile()
    model.opt.timestep = PHYSICS_DT
    data = mujoco.MjData(model)
    ctrl = None
    if bam_model is not None:
        from bam.mujoco import MujocoController

        ctrl = MujocoController(bam_model, list(HUMANOID_JOINT_NAMES), model, data)
    return model, data, ctrl, bam_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", nargs="?", help="exported policy (omit with --zero)")
    ap.add_argument("--zero", action="store_true", help="no policy: zero actions (hold HOME)")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--no-bam", action="store_true", help="XML PD actuators instead of the BAM STS3215 model")
    ap.add_argument("--video", type=Path, help="write a GIF of the run (needs MUJOCO_GL=egl or osmesa)")
    args = ap.parse_args()
    if args.onnx is None and not args.zero:
        ap.error("give an ONNX file, or --zero")

    run, meta = load_policy(None if args.zero else args.onnx)
    names = meta["joint_names"].split(",")
    assert tuple(names) == HUMANOID_JOINT_NAMES, f"policy joint order {names} != robot {HUMANOID_JOINT_NAMES}"
    assert meta["observation_names"] == "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command", meta["observation_names"]
    default = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
    scale = float(meta["action_scale"])

    model, data, bam, bam_model = build(args.no_bam)
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in names])
    act_ids = np.array([model.actuator(n).id for n in names])
    gyro = model.sensor("imu_ang_vel")

    data.qpos[2] = HUMANOID_STAND_Z + 0.005
    data.qpos[3:7] = (1, 0, 0, 0)
    data.qpos[qadr] = default
    mujoco.mj_forward(model, data)
    if bam is not None:
        bam.reset(data.qpos)
        bam_model.actuator.q_target_smooth = data.qpos[qadr].copy()

    cmd = np.array([args.vx, args.vy, args.wz])
    last = np.zeros(len(names))
    frames, renderer = [], None
    if args.video:
        renderer = mujoco.Renderer(model, 360, 480)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = model.body("pelvis").id
        cam.distance, cam.elevation, cam.azimuth = 1.8, -12, 135
    steps = int(args.seconds / (PHYSICS_DT * DECIMATION))
    start = data.qpos[:2].copy()
    yaw0 = 0.0
    fell = None
    for k in range(steps):
        R = data.xmat[model.body("pelvis").id].reshape(3, 3)
        obs = np.concatenate([
            data.sensordata[gyro.adr[0]: gyro.adr[0] + 3],
            R.T @ np.array([0.0, 0.0, -1.0]),
            data.qpos[qadr] - default,
            data.qvel[vadr],
            last,
            cmd,
        ]).astype(np.float32)
        action = np.asarray(run(obs), dtype=np.float64)
        last = action
        target = default + scale * action
        for _ in range(DECIMATION):
            if bam is not None:
                bam.q_target[:] = target
                bam.update()
            else:
                data.ctrl[act_ids] = target
            mujoco.mj_step(model, data)
        if R[2, 2] < math.cos(math.radians(60)) and fell is None:
            fell = k * PHYSICS_DT * DECIMATION
        if renderer is not None and k % 2 == 0:
            renderer.update_scene(data, cam)
            frames.append(renderer.render())

    t = steps * PHYSICS_DT * DECIMATION
    R = data.xmat[model.body("pelvis").id].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    d = data.qpos[:2] - start
    print(f"{t:.1f} s  commanded vx={args.vx:+.2f} vy={args.vy:+.2f} wz={args.wz:+.2f}")
    print(f"  moved {d[0]:+.2f} m x, {d[1]:+.2f} m y  ({np.linalg.norm(d) / t:.2f} m/s), yaw {math.degrees(yaw - yaw0):+.0f} deg")
    print(f"  pelvis z {data.qpos[2]:.3f} m, tilt {math.degrees(math.acos(max(-1, min(1, R[2, 2])))):.1f} deg, "
          + ("FELL at %.1f s" % fell if fell is not None else "stayed up"))
    if frames:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.video, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"  video -> {args.video}")


if __name__ == "__main__":
    main()
