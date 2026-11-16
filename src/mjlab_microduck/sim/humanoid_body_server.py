"""The STS3215 humanoid in MuJoCo, served to `humanoidd` (microduck repo) over TCP.

    uv run humanoid-body                  # a viewer window
    uv run humanoid-body --headless       # no window (tests, servers)

Then, on the runtime side::

    humanoidd --sim 127.0.0.1:7811 --policy policy.onnx

Same wire protocol as the duck's body server (`body_server.py`, `duck_control::sim`):
newline-delimited JSON over TCP, one request and one answer per line —
``hello`` / ``read`` / ``write`` / ``torque`` / ``gain`` / ``slow`` — in the robot's
own units (radians, rad/s, the IMU already resolved into the pelvis frame). Only
the joint table differs: 18 joints in `HUMANOID_JOINT_NAMES` order, which the
handshake checks.

The servos are the BAM STS3215 model the policy was trained against, on CPU
(`bam.mujoco.MujocoController`), at the training timestep. The robot is held in
place at HOME until the runtime turns torque on — a biped cannot wait standing
with nobody balancing it.
"""

from __future__ import annotations

import argparse
import json
import socket
import socketserver
import threading
import time

import mujoco
import numpy as np

from mjlab_microduck.robot.humanoid_constants import (
    HOME_JOINT_POS,
    HUMANOID_JOINT_NAMES,
    HUMANOID_SCENE_XML,
    HUMANOID_STAND_Z,
    BATTERY_NOMINAL_V,
    _BAM_ACTUATOR_KWARGS,
    load_sts3215_model,
)

PROTOCOL = 1
TIMESTEP = 0.005
NOMINAL_VOLTS = BATTERY_NOMINAL_V
NOMINAL_TEMP_C = 35.0
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)
# The STS3215's factory P gain; the runtime's `gain` op is a ratio against it.
KP_FW = _BAM_ACTUATOR_KWARGS["kp_fw"]


def gravity_in_frame(xmat: np.ndarray) -> np.ndarray:
    return xmat.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])


class Body:
    def __init__(self, vin: float = NOMINAL_VOLTS):
        from bam.mujoco import MujocoController

        spec = mujoco.MjSpec.from_file(str(HUMANOID_SCENE_XML))
        self.bam_model = load_sts3215_model()
        self.bam_model.actuator.kp = KP_FW
        self.bam_model.actuator.vin = vin
        limit = vin * self.bam_model.kt.value / self.bam_model.R.value
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
        self.model = spec.compile()
        self.model.opt.timestep = TIMESTEP
        self.data = mujoco.MjData(self.model)
        self.bam = MujocoController(self.bam_model, list(HUMANOID_JOINT_NAMES), self.model, self.data)
        m = self.model
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in HUMANOID_JOINT_NAMES])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in HUMANOID_JOINT_NAMES])
        self.act = np.array([m.actuator(n).id for n in HUMANOID_JOINT_NAMES])
        self.pelvis = m.body("pelvis").id
        self.gyro = m.sensor("imu_ang_vel").adr[0]
        self.lock = threading.Lock()
        self.torque_on = False
        self.released = False
        self.kp = KP_FW
        self.place()

    def place(self) -> None:
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:3] = (0.0, 0.0, HUMANOID_STAND_Z + 0.002)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[self.qadr] = [HOME_JOINT_POS[n] for n in HUMANOID_JOINT_NAMES]
        mujoco.mj_forward(self.model, d)
        self.bam.reset(d.qpos)
        self.bam.last_ts = d.time  # mj_resetData rewound the clock; bam's dt must not go negative
        self.bam_model.actuator.q_target_smooth = d.qpos[self.qadr].copy()
        self.held = d.qpos.copy()

    def reset(self) -> None:
        with self.lock:
            self.torque_on = False
            self.released = False
            self.place()

    def step(self, n: int) -> None:
        with self.lock:
            for _ in range(n):
                if self.torque_on:
                    self.bam.update()
                else:
                    self.data.ctrl[self.act] = 0.0
                mujoco.mj_step(self.model, self.data)
                if not self.released:
                    self.data.qpos[:] = self.held
                    self.data.qvel[:] = 0.0

    # ── protocol ──────────────────────────────────────────────────────────
    def sensors(self) -> dict:
        with self.lock:
            d = self.data
            pos = d.qpos[self.qadr].tolist()
            vel = d.qvel[self.vadr].tolist()
            force = d.actuator_force[self.act]
            gyro = d.sensordata[self.gyro: self.gyro + 3].tolist()
            grav = gravity_in_frame(d.xmat[self.pelvis]).tolist()
            quat = d.xquat[self.pelvis].tolist()
            pelvis = d.qpos[:3].tolist()
            t = float(d.time)
        return {
            "positions": pos,
            "velocities": vel,
            # A stand-in with the right shape, not a calibrated current: |torque| / kt.
            "currents_ma": (np.abs(force) / self.bam_model.kt.value * 1000.0).tolist(),
            "imu": {"gyro": gyro, "gravity": grav, "quat": quat},
            # Not measurable on a robot; for tools asking "is it still up?".
            "pelvis": pelvis,
            "sim_time": t,
        }

    def set_targets(self, targets) -> None:
        if len(targets) != len(HUMANOID_JOINT_NAMES):
            raise ValueError(f"expected {len(HUMANOID_JOINT_NAMES)} targets, got {len(targets)}")
        with self.lock:
            self.bam.q_target[:] = targets

    def set_torque(self, on: bool) -> None:
        with self.lock:
            if on and not self.torque_on:
                # Torque arriving must not fling the robot at a stale target.
                q = self.data.qpos[self.qadr].copy()
                self.bam.q_target[:] = q
                self.bam_model.actuator.q_target_smooth = q.copy()
                self.released = True
            self.torque_on = bool(on)

    def set_gain(self, kp: int) -> None:
        with self.lock:
            self.kp = float(kp)
            self.bam_model.actuator.kp = self.kp


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        body: Body = self.server.body
        print(f"== runtime connected from {self.client_address}", flush=True)
        for raw in self.rfile:
            try:
                answer = self.dispatch(body, json.loads(raw))
            except Exception as error:  # a bad frame must not take the simulator down
                answer = {"error": str(error)}
            self.wfile.write((json.dumps(answer) + "\n").encode())
            self.wfile.flush()
        print("== runtime disconnected", flush=True)

    def dispatch(self, body: Body, request: dict) -> dict:
        op = request.get("op")
        if op == "hello":
            if request.get("protocol") != PROTOCOL:
                raise ValueError(f"the runtime speaks protocol {request.get('protocol')}, this simulator {PROTOCOL}")
            joints = request.get("joints")
            if joints is not None and joints != len(HUMANOID_JOINT_NAMES):
                raise ValueError(f"the runtime drives {joints} joints, this humanoid has {len(HUMANOID_JOINT_NAMES)}")
            # Each new runtime finds the robot standing at HOME, held, torque off — not lying
            # wherever the last one left it.
            body.reset()
            return {"protocol": PROTOCOL, "joint_names": list(HUMANOID_JOINT_NAMES)}
        if op == "read":
            return body.sensors()
        if op == "write":
            body.set_targets(request["targets"])
            return {}
        if op == "torque":
            body.set_torque(bool(request["on"]))
            return {}
        if op == "gain":
            body.set_gain(int(request["kp"]))
            return {}
        if op == "slow":
            return {"volts": NOMINAL_VOLTS, "temps_c": [NOMINAL_TEMP_C] * len(HUMANOID_JOINT_NAMES)}
        raise ValueError(f"unknown op {op!r}")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def run(body: Body, headless: bool) -> None:
    """Real time: the runtime's loop is wall-clock, so physics must be too."""
    viewer = None
    if not headless:
        try:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(body.model, body.data, show_left_ui=False, show_right_ui=False)
        except Exception as error:
            print(f"== no viewer ({error}); running headless", flush=True)
    batch = 4  # 20 ms of world per pass = one control tick
    period = batch * TIMESTEP
    nxt = time.perf_counter()
    k = 0
    try:
        while viewer is None or viewer.is_running():
            body.step(batch)
            nxt += period
            slack = nxt - time.perf_counter()
            if slack > 0.002:
                time.sleep(slack)
            elif slack < -0.25:
                print(f"== behind real time by {-slack:.2f}s", flush=True)
                nxt = time.perf_counter()
            k += 1
            if viewer is not None and k % 2 == 0:
                with body.lock:
                    viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7811)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--vin", type=float, default=NOMINAL_VOLTS, help="battery voltage the servos see")
    args = ap.parse_args()
    body = Body(vin=args.vin)
    server = Server((args.host, args.port), Handler)
    server.body = body
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"== humanoid at HOME, held until torque is on · humanoidd --sim {args.host}:{args.port}", flush=True)
    run(body, args.headless)


if __name__ == "__main__":
    main()
