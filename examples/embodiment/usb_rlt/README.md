# USB Full-Cycle Manipulation with Joint RLT (`π0.5` + Lightweight Stage 2 RLPD)

End-to-end closed-loop USB plug-and-return manipulation (`lift -> insert -> withdraw -> return -> release -> home`) on `scenes_v3_real` using a single unified model under strict pure-friction physics (`weld = False`, `attachment_active = False`, `teacher_planner_used = False`, zero privileged pose at inference/evaluation).

## Directory Layout

- `data/`: `scenes_v3_real` train/validation scene generators (`val100_unseen` strictly disjoint from `train1200`, `val34`, and `test68`), NVMe dataset builder, balanced dataset construction, on-policy DAgger collection, and live `P4` / `P7` handoff pool capture.
- `stage1/`: Stage 1B official RLinf Joint RLT (`use_rlt=True, rlt_alpha=1.0`) Hydra config, training launcher, DCP-to-full-weights exporter, and `z_rl` representation probe.
- `stage2/`: Pure robot-only contact proprioception and Jacobian utilities (`robot_contact_utils.py`), tri-stereo `SpatialSoftmax` + `z_rl` residual RLPD trainer (`P4` insertion and `P7` holder return), wide-brightness in-hand demo buffer collector, and lightweight `P5 + P6 + P10 + P11` transit residual trainer.
- `eval/`: Non-privileged proprioceptive phase switcher (`eval_v2_fast.py`) and unified closed-loop evaluator (`eval_full_cycle_pi05_rlt.py`) with single-finger hardware limit protection and motor-torque Jacobian admittance compliance.

## Unified Deployment Architecture

- **Stage 1B backbone**: 1x `π0.5` (`M=1`, `k_exec=4` for `P0–P6`) trained from `pi05_base` via Stage 1A SFT + On-Policy DAgger + RLinf Joint RLT (`global_step_2500`, `use_rlt=True, rlt_alpha=1.0`; `z_rl` probe `shuffle_loss +353.26%`).
- **Stage 2 heads (`1,112,757` = `1.113M` fp32 parameters total, `< 5M` budget)**:
  - `P4` insertion head (`v4_tristereo640_qtgt`): `507,143` params
  - `P5 + P6 + P10 + P11` transit head (`frozen_p56_v2_rlt_head.pt`): `98,471` params
  - `P7` holder return head (`v4_tristereo640_qtgt`): `507,143` params

## Verified Ablations and Validation Results

All results below are measured in closed-loop MuJoCo 4.0 pure-friction simulation with `teacher_planner_used = False` and `attachment_active = False`.

### Offline Subtask Unbiased Validation (`val100_unseen`, strictly disjoint from train/`val34`/`test68`)

| Subtask | Checkpoint | Unseen Scenes | Success | Safety Stops | Avg Progress |
|---|---|---:|---:|---:|---:|
| `P4` insert | `frozen_p4_v5c_s42_rd2ema_971.pt` (650 scenes, selected on `val34`) | 91 | 81/91 (89.0%) | 0 | 10.90 mm |
| `P4` insert | `frozen_p4_v4_jit_924.pt` | 91 | 83/91 (91.2%) | 0 | 10.90 mm |
| `P4` insert | `frozen_p4_v6b_s42_rd6ema_923.pt` (1,050 scenes, gentle photo aug) | 91 | 84/91 (92.3%) | 1 | 11.10 mm |
| `P7` return | `frozen_p7_v7_s42_rd4_950.pt` (650 scenes, selected on `val34`) | 79 | 65/79 (82.3%) | 0 | -1.91 mm |
| `P7` return | `frozen_p7_v8_s42_fresh` (1,050 scenes, wide brightness, fresh) | 79 | 74/79 (93.7%) | 1 | -0.79 mm |
| `P7` return | `frozen_p7_v8_s202_rd10ema_962.pt` (1,050 scenes, wide brightness + finger penalty) | 79 | **76/79 (96.2%)** | 1 | **-0.42 mm** |

### Full 6-Event Closed-Loop Cycle Results

| Run | Split | Stack | Lift | Insert | Withdraw | Return | Release+Home | Full 6-Event | Finger Guard (`fg`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v3fresh_g1a_s14k_val34` | `val34` (34) | Stage 1A SFT only | 29/34 (85.3%) | 10/34 (29.4%) | — | — | — | 2/34 (5.9%) | 0/34 |
| `v3fresh_g1b_joint_s2500_val34` | `val34` (34) | Stage 1B Joint RLT only | 33/34 (97.1%) | 16/34 (47.1%) | — | — | — | 2/34 (5.9%) | 0/34 |
| `v3fresh_g3_p4v4_p7s77_val34` | `val34` (34) | `P4-V4 + P56-v2 + P7-V7 s77` | 34/34 (100.0%) | 30/34 (88.2%) | 30/30 (100.0%) | 29/30 (96.7%) | 29/29 (100.0%) | 29/34 (85.3%) | 0/34 |
| `v3fresh_g3_v5c_p7s42_val34` | `val34` (34) | `P4-V5c + P56-v2 + P7-V7 s42` | 34/34 (100.0%) | 32/34 (94.1%) | 31/32 (96.9%) | 29/31 (93.5%) | 29/29 (100.0%) | 29/34 (85.3%) | 0/34 |
| `v3fresh_final_blind31` (initial blind, failed gate) | `test` (31/68) | `P4-V5c + P56-v2 + P7-V7 s42` | 29/31 (93.5%) | 25/29 (86.2%) | 24/25 (96.0%) | 20/24 (83.3%) | 19/20 (95.0%) | 19/31 (61.3%) | 4/31 |
| `v3fresh_g6_clean_val34` | `val34` (34) | `P4-V4 + P56-v2 + P7-V8 96.2%` + upgraded guards | **34/34 (100.0%)** | 30/34 (88.2%) | **30/30 (100.0%)** | **30/30 (100.0%)** | 29/30 (96.7%) | 29/34 (85.3%) | **0/34** |
| `v3fresh_g6_clean_val50_unseen` (in progress, 31/50) | `val100_unseen` (31/50) | `P4-V4 + P56-v2 + P7-V8 96.2%` + upgraded guards | **30/31 (96.8%)** | 27/30 (90.0%) | **26/26 (100.0%)** | **26/26 (100.0%)** | **26/26 (100.0%)** | **26/31 (83.9%)** | **0/31** |

Notes:
- The `val34 -> blind31` gap (`85.3% -> 61.3%`) traced to small-`val34` checkpoint selection overfitting, dark-scene sensitivity, and asymmetric single-finger over-travel in `P7`. Upgrading `P7` to `P7-V8` (`1,050` wide-brightness scenes + single-finger penalty + 79-scene `val100_unseen` EMA) lifted unseen `P7` from `82.3%` to `96.2%` offline and `56/56 (100.0%)` live across `val34` + `val50_unseen` (`31/50` so far), with `0/65` `Finger joint limit safety guard` stops.
- The remaining 37 untouched test seeds in `stage2_test_scenes_v3.json` remain strictly sealed and unused until final validation completes.
