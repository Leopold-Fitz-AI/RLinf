# USB Full-Cycle Manipulation with Joint RLT (`π0.5` + Lightweight Stage 2 RLPD)

End-to-end closed-loop USB plug-and-return manipulation (`lift -> insert -> withdraw -> return -> release -> home`) on `scenes_v3_real` using a single unified model under strict pure-friction MuJoCo 4.0 physics (`weld = False`, `attachment_active = False`, `teacher_planner_used = False`, zero privileged pose at inference/evaluation).

## Directory Layout

- `data/`: `scenes_v3_real` train/validation scene generators (`val100_unseen` strictly disjoint from `train1200`, `val34`, and `test68`), NVMe dataset builder, balanced dataset construction, on-policy DAgger collection, and live `P4` / `P7` handoff pool capture.
- `stage1/`: Stage 1B official RLinf Joint RLT (`use_rlt=True, rlt_alpha=1.0`) Hydra config, training launcher, DCP-to-full-weights exporter, and `z_rl` representation probe.
- `stage2/`: Pure robot-only contact proprioception and Jacobian utilities (`robot_contact_utils.py`), tri-stereo `SpatialSoftmax` + `z_rl` residual RLPD trainer (`P4` insertion and `P7` holder return), wide-brightness demo buffer collectors, and lightweight `P5 + P6 + P10 + P11` transit residual trainer.
- `eval/`: Non-privileged proprioceptive phase switcher (`eval_v2_fast.py`) and unified closed-loop evaluator (`eval_full_cycle_pi05_rlt.py`) with velocity-lookahead single-finger limit protection, depth-gated rim-unjam retry, and motor-torque Jacobian admittance compliance.

## Unified Deployment Architecture

- **Stage 1B backbone**: 1x `π0.5` (`M=1`, `k_exec=4` for `P0–P6`) trained cleanly from `pi05_base` via Stage 1A SFT + On-Policy DAgger + RLinf Joint RLT (`global_step_2500`, `use_rlt=True, rlt_alpha=1.0`; `z_rl` probe `shuffle_loss +353.26%`).
- **Stage 2 heads (`1,112,757` = `1.113M` fp32 parameters total, `< 5M` budget)**:
  - `P4` insertion head (`frozen_p4_v8a_s202_rd2ema_956.pt`, zero `--inhand-rand`): `507,143` fp32 params
  - `P5 + P6 + P10 + P11` transit head (`frozen_p56_v2_rlt_head.pt`): `98,471` fp32 params
  - `P7` holder return head (`frozen_p7_v8_s202_rd10ema_962.pt`, wide brightness + single-finger penalty): `507,143` fp32 params

## Verified Ablations, Validation, and Sealed Blind Test Results

All closed-loop results below are measured in MuJoCo 4.0 pure-friction simulation with `teacher_planner_used = False`, `attachment_active = False`, and zero privileged pose.

### Offline Subtask Unbiased Validation (`val100_unseen`, strictly disjoint from train/`val34`/`test68`)

| Subtask | Checkpoint | Unseen Scenes | Success | Safety Stops | Avg Progress |
|---|---|---:|---:|---:|---:|
| `P4` insert | `frozen_p4_v5c_s42_rd2ema_971.pt` (650 scenes, selected on `val34`) | 91 | 81/91 (89.0%) | 0 | 10.90 mm |
| `P4` insert | `frozen_p4_v4_jit_924.pt` | 91 | 83/91 (91.2%) | 0 | 10.90 mm |
| `P4` insert | `frozen_p4_v6b_s42_rd6ema_923.pt` (1,050 scenes, gentle photo aug) | 91 | 84/91 (92.3%) | 1 | 11.10 mm |
| `P4` insert | **`frozen_p4_v8a_s202_rd2ema_956.pt`** (clean zero `--inhand-rand`, Rd2 EMA) | 91 | **87/91 (95.6%)** | **0** | **10.82 mm** |
| `P7` return | `frozen_p7_v7_s42_rd4_950.pt` (650 scenes, selected on `val34`) | 79 | 65/79 (82.3%) | 0 | -1.91 mm |
| `P7` return | `frozen_p7_v8_s42_fresh` (1,050 scenes, wide brightness, fresh) | 79 | 74/79 (93.7%) | 1 | -0.79 mm |
| `P7` return | **`frozen_p7_v8_s202_rd10ema_962.pt`** (1,050 scenes, wide brightness + finger penalty) | 79 | **76/79 (96.2%)** | 1 | **-0.42 mm** |

### Full 6-Event Closed-Loop Cycle Results

| Run | Split | Stack | Lift | Insert | Withdraw | Return | Release+Home | Full 6-Event | Finger Guard (`fg`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v3fresh_g1a_s14k_val34` | `val34` (34) | Stage 1A SFT only | 29/34 (85.3%) | 10/34 (29.4%) | — | — | — | 2/34 (5.9%) | 0/34 |
| `v3fresh_g1b_joint_s2500_val34` | `val34` (34) | Stage 1B Joint RLT only | 33/34 (97.1%) | 16/34 (47.1%) | — | — | — | 2/34 (5.9%) | 0/34 |
| `v3fresh_g3_v5c_p7s42_val34` | `val34` (34) | `P4-V5c + P56-v2 + P7-V7 s42` | 34/34 (100.0%) | 32/34 (94.1%) | 31/32 (96.9%) | 29/31 (93.5%) | 29/29 (100.0%) | 29/34 (85.3%) | 0/34 |
| `v3fresh_final_blind31` (initial blind, failed gate) | `test` (31/68) | `P4-V5c + P56-v2 + P7-V7 s42` | 29/31 (93.5%) | 25/29 (86.2%) | 24/25 (96.0%) | 20/24 (83.3%) | 19/20 (95.0%) | 19/31 (61.3%) | 4/31 |
| `v3fresh_final_verified_val34` | `val34` (34) | `P4-V8a + P56-v2 + P7-V8` + `phase_step` + `fq_slam_guard` | 34/34 (100.0%) | 32/34 (94.1%) | 31/31 (100.0%) | 31/31 (100.0%) | 31/31 (100.0%) | **31/34 (91.2%)** | **0/34** |
| `v3fresh_frozen_gate_val50_unseen` | `val100_unseen` (first 50) | `P4-V8a + P56-v2 + P7-V8` (`max_steps=2800`) | 48/50 (96.0%) | 44/48 (91.7%) | 43/43 (100.0%) | 41/43 (95.3%) | 41/41 (100.0%) | 41/50 (82.0%) | **0/50** |
| `v3fresh_frozen_holdout_val50b` (untouched unseen holdout) | `val100_unseen` (other 50) | **Final frozen stack** (`p4_above_rim` + `5.5 N` compliance, `max_steps=3600`) | **50/50 (100.0%)** | **47/50 (94.0%)** | **46/46 (100.0%)** | **45/46 (97.8%)** | **45/45 (100.0%)** | **45/50 (90.0%)** | **0/50** |
| **`v3fresh_final_pristine_blind37` (one-shot sealed blind test)** | **`test` (remaining 37/68 untouched)** | **Final frozen stack (byte-identical to `val50b`, `--save-video`)** | **37/37 (100.0%)** | **36/37 (97.3%)** | **36/36 (100.0%)** | **33/36 (91.7%)** | **33/33 (100.0%)** | **33/37 (89.2%)** | **0/37** |

### Key Engineering Findings

1. **`--inhand-rand` helps `P7` (`+13.9%` to `96.2%`) but degrades `P4`**: `P7` runs after `P4+P5` contact where genuine in-hand slip exists and the holder slot is wide (`±2 mm`), whereas `P4` follows a clean vertical grasp (`<0.5 mm` slip) into a `±0.3 mm` metal socket. Removing `--inhand-rand` on `P4` cut auxiliary geometry error by `3.3x` and lifted unseen `P4` to `87/91 (95.6%, safety=0)`.
2. **Phase-local `M=1` flow-matching seed alignment (`step_seed = 7317 + phase_step`)**: Matching live phase-entry noise seeds to offline Stage 2 training closed a `-12%` offline-to-live `P4` gap immediately (`insert` rose to `94.0%–97.3%` live across `val50b` and `blind37`).
3. **Single-finger velocity lookahead (`fq_slam_guard`) + depth-gated rim-unjam (`p4_above_rim`)**: Predicting 1-step empty-air finger slams (`fq_min_vel < -0.0005` and `fq_min_now + fq_min_vel < 0.00010`) eliminated `Finger joint limit safety guard` completely (`0/171` across `val34`, `val50a`, `val50b`, and `blind37`), while restricting `P4` rim-unjam to `p4_entry_z - tcp_z < 0.015` eliminated upward yanks after seating (`0/37` insertion force stops on `blind37`).
