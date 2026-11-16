# STS3215 humanoid (18 DoF)

A second robot trained in this repo: an ~825 mm biped of 18 Feetech STS3215 servos, generated from
the frame CAD (`robot/humanoid/build_model.py`). Its runtime is `humanoidd` in the `microduck`
repo (`docs/design/humanoid.md` there owns the joint contract and bring-up).

It does **not** share the duck's 61-D obs contract: it is its own policy family
(63-D obs = gyro 3, gravity 3, q−HOME 18, q̇ 18, last action 18, twist 3; 18-D action, scale 0.5).

```bash
uv run train Mjlab-Velocity-Flat-Humanoid --env.scene.num-envs 4096     # GPU, or add --hf-jobs
uv run train Mjlab-Velocity-Rough-Humanoid --env.scene.num-envs 4096
uv run scripts/export.py Mjlab-Velocity-Flat-Humanoid --wandb-run-path <...>
uv run scripts/infer_humanoid.py policy.onnx --vx 0.3 --seconds 10     # CPU rehearsal (BAM actuators)
uv run humanoid-body --headless                                        # body for humanoidd --sim
uv run --with pytest pytest tests/test_humanoid_cfg.py
```

Regenerating the model from new STLs:

```bash
uv run --with trimesh --with fast-simplification python \
  src/mjlab_microduck/robot/humanoid/build_model.py --stl-dir <frame>/stl/stage2_lightweight
```

Assumptions baked into the model (`mass_report.json` lists each): PLA at 60 % effective fill,
55 g per servo, Pi 5-class computer, 3S 2200 mAh LiPo (185 g), LD19-class LiDAR, Pi Camera-class
camera, IMU in the pelvis.

## Servos: 12 V STS3215 on 3S

bam ships a fit of the 7.4 V STS3215 only. `robot/humanoid/sts3215_12V_derived.json` scales it by
Feetech's datasheet ratios — stall 30 kg·cm at 12 V against 19.5 kg·cm at 7.4 V, and the same
no-load speed (0.222 s/60°) at each one's rated voltage — keeping friction and armature. It is
**not fitted**. Note the 7.4 V fit is itself ~1.75× stronger at stall than its datasheet, and the
derived 12 V model inherits that: ~5.0 N·m at 11.7 V against a datasheet 2.94 N·m. Refit with
`bam` (`bam/feetech/record.py`) on a real 12 V servo and point `STS3215_PARAMS_JSON` at the result.

`uv run scripts/humanoid_torque_budget.py [--knee 1.0]` prints each joint's static torque standing
on one foot (CoP at the foot centre and at the sole's edges) against the datasheet stall. At HOME
the stance knee (68 % worst case, heel CoP) and ankle (52 %, toe) are the tight ones. Collision shapes are boxes; feet are flat box soles.
The ankle has pitch only (no roll), as in the frame.
