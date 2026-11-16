"""Static joint torques of the humanoid standing on one foot, against the servo's limits.

    uv run scripts/humanoid_torque_budget.py               # HOME pose
    uv run scripts/humanoid_torque_budget.py --knee 0.9    # a deeper crouch (hip/ankle follow)

Single support, static: the whole weight comes up through the stance foot at a centre of
pressure (CoP) — the foot centre, and the sole's edges (where a balancing robot actually
pushes). For each stance-leg joint, the torque is the moment of that ground reaction about
the joint axis, less the weight of the links between the joint and the floor. Swing-leg and
upper-body joints just hold their own distal links. No dynamics: walking adds on top.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from mjlab_microduck.robot.humanoid_constants import (
    BATTERY_NOMINAL_V,
    HOME_JOINT_POS,
    HUMANOID_JOINT_NAMES,
    HUMANOID_SCENE_XML,
    load_sts3215_model,
)

G = 9.81
DATASHEET_STALL_NM = 30.0 * 0.0980665  # 12 V STS3215: 30 kg·cm at 12 V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knee", type=float, default=None, help="stance knee angle (rad); hip and ankle take -knee/2")
    ap.add_argument("--vin", type=float, default=BATTERY_NOMINAL_V)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(HUMANOID_SCENE_XML))
    d = mujoco.MjData(m)
    q = dict(HOME_JOINT_POS)
    if args.knee is not None:
        for side, s in (("left", 1.0), ("right", -1.0)):
            q[f"{side}_knee"] = s * args.knee
            q[f"{side}_hip_pitch"] = q[f"{side}_ankle"] = -s * args.knee / 2
    for n in HUMANOID_JOINT_NAMES:
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = q[n]
    d.qpos[3:7] = (1, 0, 0, 0)
    mujoco.mj_forward(m, d)

    total = float(m.body_subtreemass[m.body("pelvis").id])
    foot = m.body("foot_L").id
    sole = m.geom("left_foot_collision").id
    centre = d.geom_xpos[sole].copy()
    half = m.geom_size[sole][:2]
    rot = d.geom_xmat[sole].reshape(3, 3)
    cops = {
        "centre": centre,
        "toe": centre + rot @ np.array([half[0], 0, 0]),
        "heel": centre - rot @ np.array([half[0], 0, 0]),
        "inner edge": centre + rot @ np.array([0, -np.sign(centre[1]) * half[1], 0]),
        "outer edge": centre + rot @ np.array([0, np.sign(centre[1]) * half[1], 0]),
    }
    floor_z = centre[2] - m.geom_size[sole][2]
    for k in cops:
        cops[k][2] = floor_z

    bam = load_sts3215_model()
    stall_sim = args.vin * bam.kt.value / bam.R.value

    def chain_to(body: int) -> list[int]:
        out = []
        while body > 0:
            out.append(body)
            body = m.body_parentid[body]
        return out

    stance_chain = set(chain_to(foot))
    rows = []
    for n in HUMANOID_JOINT_NAMES:
        j = m.joint(n).id
        b = m.jnt_bodyid[j]
        a, ax = d.xanchor[j], d.xaxis[j]
        sub = [k for k in range(1, m.nbody) if b in chain_to(k)]
        own = sum(np.cross(d.xipos[k] - a, [0, 0, -m.body_mass[k] * G]) for k in sub)
        if b in stance_chain:
            per = {name: abs(np.dot(np.cross(p - a, [0, 0, total * G]) + own, ax)) for name, p in cops.items()}
        else:
            per = {"centre": abs(np.dot(own, ax))}
        rows.append((n, per["centre"], max(per.values()), max(per, key=per.get)))

    print(f"mass {total:.2f} kg · single support on the left foot · knee {q['left_knee']:+.2f} rad")
    print(f"limits: sim model stall {stall_sim:.2f} N·m at {args.vin:.1f} V · datasheet stall {DATASHEET_STALL_NM:.2f} N·m at 12 V")
    print(f"{'joint':22s} {'CoP centre':>10s} {'worst CoP':>10s}  {'(where)':12s} {'% datasheet':>11s}")
    for n, c, w, where in rows:
        flag = "  <-- over 50 %" if w > 0.5 * DATASHEET_STALL_NM else ""
        print(f"{n:22s} {c:10.2f} {w:10.2f}  {where:12s} {100 * w / DATASHEET_STALL_NM:10.0f}%{flag}")


if __name__ == "__main__":
    main()
