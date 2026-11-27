#!/usr/bin/env bash
set -euo pipefail

# Strict GPU 4-7 ONLY 4-shard full-cycle closed-loop evaluator.
# Usage: bash run_eval_4gpu_gpu47.sh <LABEL> <SPLIT> <LIMIT> <SAVE_VIDEO:0|1> <P4_CKPT> <P7_CKPT>
LABEL="${1:?LABEL required}"
SPLIT="${2:?SPLIT required (validation|test)}"
LIMIT="${3:?LIMIT required}"
SAVE_VIDEO="${4:-0}"
P4_CKPT="${5:?P4_CKPT required}"
P7_CKPT="${6:?P7_CKPT required}"

H2=/vla-cd/jy/v12b_usb_cycle_v1/holder_v2_rlt
PY=/root/g1_usb_sim2real/.venv-pi05/bin/python
S1_CKPT=$H2/stage1_pi05_rlt/pi05_v3_fresh_joint_rlt_s1/pi05_v3_fresh_joint_rlt_s1/checkpoints/global_step_2500/actor/dcp_checkpoint
P56_CKPT=$H2/stage2_rlt_opt3a/frozen_p56_v2_rlt_head.pt
LOG_DIR=$H2/logs/$LABEL

mkdir -p "$LOG_DIR"
export PYTHONPATH="/root/v12b_usb_cycle_v1/runtime:$H2/code:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase12_v1/code:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/code:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/dependencies:/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/rlinf:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

test -f "$P4_CKPT"
test -f "$P56_CKPT"
test -f "$P7_CKPT"

EXTRA_ARGS=()
if [ "$SAVE_VIDEO" = "1" ]; then
  EXTRA_ARGS+=(--save-video)
fi

echo "[$(date '+%F %T')] === Launch 4-GPU Full-Cycle Eval: label=$LABEL split=$SPLIT limit=$LIMIT save_video=$SAVE_VIDEO ==="
echo "  S1=$S1_CKPT"
echo "  P4=$P4_CKPT"
echo "  P56=$P56_CKPT"
echo "  P7=$P7_CKPT"

for shard in 0 1 2 3; do
  gpu=$((4 + shard))
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu \
    $PY -u "$H2/code/eval_full_cycle_pi05_rlt.py" \
      --label "$LABEL" \
      --stage1-config-dir "$H2/configs" \
      --stage1-config-name pi05_v3_fresh_joint_rlt_s1 \
      --stage1-ckpt "$S1_CKPT" \
      --rlt-adapter-pt "" \
      --p4-rlt-ckpt "$P4_CKPT" \
      --p56-rlt-ckpt "$P56_CKPT" \
      --p7-rlt-ckpt "$P7_CKPT" \
      --scenes-root "$H2/scenes_v3_real" \
      --split "$SPLIT" \
      --limit "$LIMIT" \
      --phase-flow-m "4:1" \
      --num-shards 4 \
      --shard-idx "$shard" \
      "${EXTRA_ARGS[@]}" \
      > "$LOG_DIR/shard_${shard}_gpu${gpu}.log" 2>&1 &
done

wait
echo "[$(date '+%F %T')] === ALL 4 SHARDS COMPLETED FOR $LABEL ==="
