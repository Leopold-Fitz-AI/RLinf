#!/usr/bin/env bash
set -euo pipefail

# Strict GPU 4-7 ONLY pipeline for P7-V7:
# - Train-only physical in-hand plug slip + tilt randomization (destroys gripper-TCP/wrist-center shortcut!)
# - Smooth continuous descent gate (--smooth-p7, verified 29/29=100% saf=0 med 52 steps, zero Y-axis limit cycle!)
# - Anti-shortcut proprio & prev_act6 dropout + 4x XY/rot aux_geom gradient boost!
H2=/vla-cd/jy/v12b_usb_cycle_v1/holder_v2_rlt
PY=/root/g1_usb_sim2real/.venv-pi05/bin/python
S1_CKPT=$H2/stage1_pi05_rlt/pi05_v3_fresh_joint_rlt_s1/pi05_v3_fresh_joint_rlt_s1/checkpoints/global_step_2500/actor/dcp_checkpoint
VAL_POOL=$H2/handoff_pools_v3_fresh/p7_p56_pool_validation.json
TR_POOL=$H2/handoff_pools_v3_fresh/p7_p56_pool_train.json
BUF_PT=/mlplatform/jy/v3_fresh_nvme/stage2_buffers/p7_v7_inhand_650_demo_buf.pt
LOG_DIR=$H2/logs/p7_v7_inhand
OUT_BASE=$H2/stage2_rlt_opt3a

mkdir -p "$LOG_DIR" "$(dirname "$BUF_PT")"
export PYTHONPATH="/root/v12b_usb_cycle_v1/runtime:$H2/code:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase12_v1/code:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/code:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/dependencies:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/rlinf:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "[$(date '+%F %T')] === STAGE 1: Build 650-scene In-Hand Slip+Tilt Smooth-P7 Tri-Stereo Buffer (GPUs 4-7 EGL) ==="
CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 STAGE2_EGL_GPUS=4,5,6,7 \
  $PY -u "$H2/code/collect_rlt_stage2_buffer.py" \
    --subtask p7_holder --p7-q2lift --p7-inhand-rand \
    --stage1-config-dir "$H2/configs" --stage1-config-name pi05_v3_fresh_joint_rlt_s1 \
    --stage1-ckpt "$S1_CKPT" --rlt-adapter-pt "" \
    --scenes-root "$H2/scenes_v3_real" \
    --train-pool-json "$TR_POOL" \
    --val-pool-json "$VAL_POOL" \
    --out-pt "$BUF_PT" \
    --workers 12 --batch-size 16 --max-steps 220 --vla-stride 6 \
    --scale-xyz-mm 1.8 --scale-rot-rad 0.022 --q-clamp-rad 0.018 \
    --max-all-scenes 650 \
    --hires-wrist-roi --hires-front-roi --hires-left-roi \
    --exact-nullspace --soft-p7 --smooth-p7 \
    > "$LOG_DIR/build_p7_v7_buf.log" 2>&1

echo "[$(date '+%F %T')] === STAGE 2: Launch 4x P7-V7 In-Hand Vision-Forced RLPD runs across GPUs 4, 5, 6, 7 ==="
COMMON_ARGS=(
  --subtask p7_holder --p7-q2lift --p7-inhand-rand
  --stage1-config-dir "$H2/configs"
  --stage1-config-name pi05_v3_fresh_joint_rlt_s1
  --stage1-ckpt "$S1_CKPT"
  --rlt-adapter-pt ""
  --scenes-root "$H2/scenes_v3_real"
  --use-all-train-scenes
  --train-pool-json "$TR_POOL"
  --val-pool-json "$VAL_POOL"
  --preload-buffer-pt "$BUF_PT"
  --rounds 12 --eps-per-round 16 --updates-per-round 400
  --max-steps 220 --flow-m 1 --vla-stride 6
  --scale-xyz-mm 1.8 --scale-rot-rad 0.022 --q-clamp-rad 0.018
  --batch-size 128 --val-fast-limit 12
  --hires-wrist-roi --hires-front-roi --hires-left-roi
  --exact-nullspace --soft-p7 --smooth-p7 --aux-xyrot-boost
  --ema-eval --no-shift-aug --eval-jitter-val
)

CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 \
  $PY -u "$H2/code/rlt_stage2_residual_ac.py" "${COMMON_ARGS[@]}" \
    --pretrain-steps 7000 --init-sigma 0.13 \
    --proprio-dropout-prob 0.25 --prev-act-drop-prob 0.75 --ref-dropout-prob 0.30 \
    --real-tr-prob 0.50 --real-jitter-prob 0.40 --q-weight-max 0.03 --aux-geom-weight 3.0 \
    --seed 42 --out-dir "$OUT_BASE/p7_v7_ih_s42" \
    > "$LOG_DIR/p7_v7_ih_s42.log" 2>&1 &

CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 \
  $PY -u "$H2/code/rlt_stage2_residual_ac.py" "${COMMON_ARGS[@]}" \
    --pretrain-steps 7000 --init-sigma 0.13 \
    --proprio-dropout-prob 0.35 --prev-act-drop-prob 0.85 --ref-dropout-prob 0.35 \
    --real-tr-prob 0.45 --real-jitter-prob 0.35 --q-weight-max 0.02 --aux-geom-weight 3.5 \
    --seed 77 --out-dir "$OUT_BASE/p7_v7_ih_s77" \
    > "$LOG_DIR/p7_v7_ih_s77.log" 2>&1 &

CUDA_VISIBLE_DEVICES=6 MUJOCO_EGL_DEVICE_ID=6 \
  $PY -u "$H2/code/rlt_stage2_residual_ac.py" "${COMMON_ARGS[@]}" \
    --pretrain-steps 8000 --init-sigma 0.12 \
    --proprio-dropout-prob 0.20 --prev-act-drop-prob 0.70 --ref-dropout-prob 0.25 \
    --real-tr-prob 0.55 --real-jitter-prob 0.45 --q-weight-max 0.00 --aux-geom-weight 3.0 \
    --seed 202 --out-dir "$OUT_BASE/p7_v7_ih_s202" \
    > "$LOG_DIR/p7_v7_ih_s202.log" 2>&1 &

CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 \
  $PY -u "$H2/code/rlt_stage2_residual_ac.py" "${COMMON_ARGS[@]}" \
    --pretrain-steps 7500 --init-sigma 0.12 \
    --proprio-dropout-prob 0.40 --prev-act-drop-prob 1.00 --ref-dropout-prob 0.40 \
    --real-tr-prob 0.40 --real-jitter-prob 0.35 --q-weight-max 0.02 --aux-geom-weight 4.0 \
    --seed 99 --out-dir "$OUT_BASE/p7_v7_ih_s99" \
    > "$LOG_DIR/p7_v7_ih_s99.log" 2>&1 &

wait
echo "[$(date '+%F %T')] === ALL P7-V7 IN-HAND PIPELINE STAGES COMPLETED ==="
