#!/usr/bin/env bash
set -euo pipefail

# Strict GPU 4-7 ONLY pipeline for P4-V5c Surgical Low-LR Warm-Start:
# - Evaluates on EXACT LIVE 34-scene P3->P4 handoff pool (p4_pool_validation_live34.json)
#   with --p4-extra-hold-steps 18 (guarantees ZERO post-insert 25N bottom-out or pullout!)
# - Warm-starts from frozen_p4_v4_jit_924.pt (30/34=88.2% live, 0/34 fg) on ALREADY-BUILT
#   p4_v5_inhand_650_demo_buf.pt (43,446 in-hand slip+tilt transitions)
# - ZERO proprio_dropout (0.0) across all 4 GPUs + surgical LR (4e-5..7e-5) to preserve
#   all 30/34 P4-V4 scenes while rescuing the 4 in-hand Y-slipped scenes (610230/612173/612317/612318)!
H2=/vla-cd/jy/v12b_usb_cycle_v1/holder_v2_rlt
PY=/root/g1_usb_sim2real/.venv-pi05/bin/python
S1_CKPT=$H2/stage1_pi05_rlt/pi05_v3_fresh_joint_rlt_s1/pi05_v3_fresh_joint_rlt_s1/checkpoints/global_step_2500/actor/dcp_checkpoint
INIT_CKPT=$H2/stage2_rlt_opt3a/frozen_p4_v4_jit_924.pt
VAL_POOL=$H2/handoff_pools_v3_live34/p4_pool_validation_live34.json
TR_POOL=$H2/handoff_pools_v3_fresh/p4_pool_train.json
BUF_PT=/mlplatform/jy/v3_fresh_nvme/stage2_buffers/p4_v5_inhand_650_demo_buf.pt
LOG_DIR=$H2/logs/p4_v5c_live34
OUT_BASE=$H2/stage2_rlt_opt3a

mkdir -p "$LOG_DIR"
export PYTHONPATH="/root/v12b_usb_cycle_v1/runtime:$H2/code:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase12_v1/code:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/code:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/dependencies:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/rlinf:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

test -f "$INIT_CKPT"
test -f "$BUF_PT"
test -f "$VAL_POOL"

echo "[$(date '+%F %T')] === Launch 4x P4-V5c Surgical Low-LR Live34 Warm-Start across GPUs 4, 5, 6, 7 ==="

COMMON_ARGS=(
  --subtask p4_insert --inhand-rand
  --stage1-config-dir "$H2/configs"
  --stage1-config-name pi05_v3_fresh_joint_rlt_s1
  --stage1-ckpt "$S1_CKPT"
  --rlt-adapter-pt ""
  --init-ckpt "$INIT_CKPT"
  --scenes-root "$H2/scenes_v3_real"
  --use-all-train-scenes
  --train-pool-json "$TR_POOL"
  --val-pool-json "$VAL_POOL"
  --preload-buffer-pt "$BUF_PT"
  --pretrain-steps 0
  --rounds 8 --eps-per-round 16 --updates-per-round 450
  --max-steps 260 --p4-extra-hold-steps 18
  --flow-m 1 --vla-stride 6
  --scale-xyz-mm 1.8 --scale-rot-rad 0.022 --q-clamp-rad 0.020
  --batch-size 128 --val-fast-limit 12
  --hires-wrist-roi --hires-front-roi --hires-left-roi
  --exact-nullspace --noback-p4 --aux-xyrot-boost
  --ema-eval --no-shift-aug --eval-jitter-val
)

# GPU 4: Ultra-gentle 5e-5, 0% dropout everywhere
CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 \
  $PY -u "$H2/code/rlt_stage2_residual_ac.py" "${COMMON_ARGS[@]}" \
    --lr-hi 5e-5 --init-sigma 0.06 \
    --proprio-dropout-prob 0.0 --prev-act-drop-prob 0.0 --ref-dropout-prob 0.0 \
    --real-tr-prob 0.60 --real-jitter-prob 0.65 --q-weight-max 0.025 --aux-geom-weight 2.0 \
    --seed 42 --out-dir "$OUT_BASE/p4_v5c_live_s42" \
    > "$LOG_DIR/p4_v5c_live_s42.log" 2>&1 &

# GPU 5: Gentle 6e-5, 0% proprio/prev_act drop, 15% ref drop, aux_geom=2.5
CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 \
  $PY -u "$H2/code/rlt_stage2_residual_ac.py" "${COMMON_ARGS[@]}" \
    --lr-hi 6e-5 --init-sigma 0.07 \
    --proprio-dropout-prob 0.0 --prev-act-drop-prob 0.0 --ref-dropout-prob 0.15 \
    --real-tr-prob 0.55 --real-jitter-prob 0.60 --q-weight-max 0.020 --aux-geom-weight 2.5 \
    --seed 77 --out-dir "$OUT_BASE/p4_v5c_live_s77" \
    > "$LOG_DIR/p4_v5c_live_s77.log" 2>&1 &

# GPU 6: 7e-5, 0% proprio drop, 10% prev_act, 15% ref, q_w=0.03, aux_geom=2.5
CUDA_VISIBLE_DEVICES=6 MUJOCO_EGL_DEVICE_ID=6 \
  $PY -u "$H2/code/rlt_stage2_residual_ac.py" "${COMMON_ARGS[@]}" \
    --lr-hi 7e-5 --init-sigma 0.07 \
    --proprio-dropout-prob 0.0 --prev-act-drop-prob 0.10 --ref-dropout-prob 0.15 \
    --real-tr-prob 0.55 --real-jitter-prob 0.65 --q-weight-max 0.030 --aux-geom-weight 2.5 \
    --seed 202 --out-dir "$OUT_BASE/p4_v5c_live_s202" \
    > "$LOG_DIR/p4_v5c_live_s202.log" 2>&1 &

# GPU 7: Most conservative 4e-5, 0% proprio/prev_act, 10% ref, real_tr=0.65/jitter=0.70
CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 \
  $PY -u "$H2/code/rlt_stage2_residual_ac.py" "${COMMON_ARGS[@]}" \
    --lr-hi 4e-5 --init-sigma 0.05 \
    --proprio-dropout-prob 0.0 --prev-act-drop-prob 0.0 --ref-dropout-prob 0.10 \
    --real-tr-prob 0.65 --real-jitter-prob 0.70 --q-weight-max 0.020 --aux-geom-weight 2.0 \
    --seed 123 --out-dir "$OUT_BASE/p4_v5c_live_s123" \
    > "$LOG_DIR/p4_v5c_live_s123.log" 2>&1 &

wait
echo "[$(date '+%F %T')] === ALL P4-V5c SURGICAL LIVE34 RUNS COMPLETED ==="
