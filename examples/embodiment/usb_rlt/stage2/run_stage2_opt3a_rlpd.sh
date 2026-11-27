#!/usr/bin/env bash
set -euo pipefail
H2=/vla-cd/jy/v12b_usb_cycle_v1/holder_v2_rlt
EXP12=/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase12_v1
EXP13=/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1
PY=/root/g1_usb_sim2real/.venv-pi05/bin/python

GPU_IDS_CSV="${GPU_IDS:-4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS_CSV"
NSHARDS="${#GPU_ARR[@]}"
G0="${GPU_ARR[0]}"
G1="${GPU_ARR[1]:-$G0}"
G2="${GPU_ARR[2]:-$G0}"
G3="${GPU_ARR[3]:-$G1}"

S1B_STEP="${S1B_STEP:-2500}"
S1B_CFG="${S1B_CFG:-pi05_v3_fresh_joint_rlt_s1}"
S1B_CKPT="${S1B_CKPT:-$H2/stage1_pi05_rlt/pi05_v3_fresh_joint_rlt_s1/pi05_v3_fresh_joint_rlt_s1/checkpoints/global_step_${S1B_STEP}/actor/dcp_checkpoint}"
POOL_DIR="$H2/handoff_pools_v3_fresh"
BUF_DIR="/mlplatform/jy/v3_fresh_nvme/stage2_buffers"
RLPD_OUT="$H2/stage2_rlt_opt3a"

export PYTHONPATH="/root/v12b_usb_cycle_v1/runtime:$H2/code:$EXP12/code:$EXP13/code:$EXP13/dependencies:$EXP13/rlinf:/vla-cd/jy/v12b_usb_cycle_v1:/vla-cd/jy/v12b_usb_cycle_v1/approach_audit_v1:/vla-cd/jy/v12b_usb_cycle_v1/revisions/teacher_axis_return_v1:/root/v12b_usb_cycle_v1/base_code/scripts:/root/v12b_usb_cycle_v1/base_code/src"
export JAX_PLATFORMS=cpu
export TORCH_COMPILE_DISABLE=1
export HF_HOME="$H2/.cache/hf"
export TORCH_HOME="$H2/.cache/torch"
export XDG_CACHE_HOME="$H2/.cache/xdg"

mkdir -p "$POOL_DIR" "$BUF_DIR" "$RLPD_OUT"

echo "=== [STAGE 2-1 REAL HANDOFF POOLS $(date)] GPUS=$GPU_IDS_CSV CKPT=$S1B_CKPT ==="
# Val34 on G0, Train 240 across remaining GPUs (or all 4 shards)
CUDA_VISIBLE_DEVICES=$G0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G0 \
  "$PY" -u "$H2/code/collect_real_handoff_pools.py" \
    --config-dir "$H2/configs" --config-name "$S1B_CFG" --checkpoint "$S1B_CKPT" \
    --split validation --limit 34 --scenes-root "$H2/scenes_v3_real" --out-dir "$POOL_DIR" \
    > "$H2/logs/s2_pool_val34.log" 2>&1 &
PID_VAL=$!

TRAIN_PIDS=()
for i in 0 1 2; do
  G="${GPU_ARR[$((i + 1))]}"
  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G \
    "$PY" -u "$H2/code/collect_real_handoff_pools.py" \
      --config-dir "$H2/configs" --config-name "$S1B_CFG" --checkpoint "$S1B_CKPT" \
      --split train --limit 240 --shard-idx "$i" --num-shards 3 \
      --scenes-root "$H2/scenes_v3_real" --out-dir "$POOL_DIR" \
      > "$H2/logs/s2_pool_train_s${i}.log" 2>&1 &
  TRAIN_PIDS+=($!)
done
wait "$PID_VAL"
for pid in "${TRAIN_PIDS[@]}"; do wait "$pid"; done

# Merge train pool shards into single JSONs
"$PY" -c "
import json, glob
from pathlib import Path
d = Path('$POOL_DIR')
for sub in ('p4_pool', 'p7_pool'):
    items = []
    for f in sorted(d.glob(f'{sub}_train_shard*.json')):
        items.extend(json.loads(f.read_text()))
    (d / f'{sub}_train.json').write_text(json.dumps(items, indent=2) + '\n')
    print(f'[MERGED POOL] {sub}_train.json: {len(items)} items')
"

echo "=== [STAGE 2-2 OPTION 3A HIRES-WRIST DEMO BUFFERS $(date)] ==="
STAGE2_EGL_GPUS="$G0,$G2" CUDA_VISIBLE_DEVICES=$G0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G0 \
  "$PY" -u "$H2/code/collect_rlt_stage2_buffer.py" \
    --subtask p4_insert \
    --stage1-config-dir "$H2/configs" --stage1-config-name "$S1B_CFG" --stage1-ckpt "$S1B_CKPT" \
    --rlt-adapter-pt "" --scenes-root "$H2/scenes_v3_real" \
    --train-pool-json "$POOL_DIR/p4_pool_train.json" --val-pool-json "$POOL_DIR/p4_pool_validation.json" \
    --out-pt "$BUF_DIR/p4_opt3a_demo_buf.pt" \
    --workers 16 --batch-size 16 --max-steps 220 --vla-stride 6 --max-all-scenes 500 --hires-wrist-roi \
    > "$H2/logs/s2_buf_p4_opt3a.log" 2>&1 &
PID_B4=$!

STAGE2_EGL_GPUS="$G1,$G3" CUDA_VISIBLE_DEVICES=$G1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G1 \
  "$PY" -u "$H2/code/collect_rlt_stage2_buffer.py" \
    --subtask p7_holder \
    --stage1-config-dir "$H2/configs" --stage1-config-name "$S1B_CFG" --stage1-ckpt "$S1B_CKPT" \
    --rlt-adapter-pt "" --scenes-root "$H2/scenes_v3_real" \
    --train-pool-json "$POOL_DIR/p7_pool_train.json" --val-pool-json "$POOL_DIR/p7_pool_validation.json" \
    --out-pt "$BUF_DIR/p7_opt3a_demo_buf.pt" \
    --workers 16 --batch-size 16 --max-steps 220 --vla-stride 6 --max-all-scenes 500 --hires-wrist-roi \
    > "$H2/logs/s2_buf_p7_opt3a.log" 2>&1 &
PID_B7=$!
wait "$PID_B4"
wait "$PID_B7"

echo "=== [STAGE 2-3 PARALLEL RLPD SWEEP (2x P4 + 2x P7, OPTION 3A) $(date)] ==="
CUDA_VISIBLE_DEVICES=$G0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G0 \
  "$PY" -u "$H2/code/rlt_stage2_residual_ac.py" \
    --subtask p4_insert \
    --stage1-config-dir "$H2/configs" --stage1-config-name "$S1B_CFG" --stage1-ckpt "$S1B_CKPT" \
    --rlt-adapter-pt "" --scenes-root "$H2/scenes_v3_real" --use-all-train-scenes \
    --train-pool-json "$POOL_DIR/p4_pool_train.json" --val-pool-json "$POOL_DIR/p4_pool_validation.json" \
    --preload-buffer-pt "$BUF_DIR/p4_opt3a_demo_buf.pt" --pretrain-steps 4500 \
    --rounds 24 --eps-per-round 16 --updates-per-round 400 --max-steps 220 \
    --vla-stride 6 --scale-xyz-mm 1.8 --scale-rot-rad 0.022 --q-weight-max 0.06 \
    --hires-wrist-roi --seed 42 --out-dir "$RLPD_OUT/p4_s42_xyz18_qw06" \
    > "$H2/logs/s2_rlpd_p4_a.log" 2>&1 &
PID_R0=$!

STAGE2_EGL_GPUS="$G1,$G3" CUDA_VISIBLE_DEVICES=$G1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G1 \
  "$PY" -u "$H2/code/rlt_stage2_residual_ac.py" \
    --subtask p4_insert \
    --stage1-config-dir "$H2/configs" --stage1-config-name "$S1B_CFG" --stage1-ckpt "$S1B_CKPT" \
    --rlt-adapter-pt "" --scenes-root "$H2/scenes_v3_real" --use-all-train-scenes \
    --train-pool-json "$POOL_DIR/p4_pool_train.json" --val-pool-json "$POOL_DIR/p4_pool_validation.json" \
    --preload-buffer-pt "$BUF_DIR/p4_opt3a_demo_buf.pt" --pretrain-steps 4500 \
    --rounds 24 --eps-per-round 16 --updates-per-round 400 --max-steps 220 \
    --vla-stride 6 --scale-xyz-mm 2.2 --scale-rot-rad 0.025 --q-weight-max 0.10 \
    --hires-wrist-roi --seed 101 --out-dir "$RLPD_OUT/p4_s101_xyz22_qw10" \
    > "$H2/logs/s2_rlpd_p4_b.log" 2>&1 &
PID_R1=$!

CUDA_VISIBLE_DEVICES=$G2 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G2 \
  "$PY" -u "$H2/code/rlt_stage2_residual_ac.py" \
    --subtask p7_holder \
    --stage1-config-dir "$H2/configs" --stage1-config-name "$S1B_CFG" --stage1-ckpt "$S1B_CKPT" \
    --rlt-adapter-pt "" --scenes-root "$H2/scenes_v3_real" --use-all-train-scenes \
    --train-pool-json "$POOL_DIR/p7_pool_train.json" --val-pool-json "$POOL_DIR/p7_pool_validation.json" \
    --preload-buffer-pt "$BUF_DIR/p7_opt3a_demo_buf.pt" --pretrain-steps 4500 \
    --rounds 24 --eps-per-round 16 --updates-per-round 400 --max-steps 220 \
    --vla-stride 6 --scale-xyz-mm 1.8 --scale-rot-rad 0.022 --q-weight-max 0.06 \
    --hires-wrist-roi --seed 42 --out-dir "$RLPD_OUT/p7_s42_xyz18_qw06" \
    > "$H2/logs/s2_rlpd_p7_a.log" 2>&1 &
PID_R2=$!

CUDA_VISIBLE_DEVICES=$G3 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G3 \
  "$PY" -u "$H2/code/rlt_stage2_residual_ac.py" \
    --subtask p7_holder \
    --stage1-config-dir "$H2/configs" --stage1-config-name "$S1B_CFG" --stage1-ckpt "$S1B_CKPT" \
    --rlt-adapter-pt "" --scenes-root "$H2/scenes_v3_real" --use-all-train-scenes \
    --train-pool-json "$POOL_DIR/p7_pool_train.json" --val-pool-json "$POOL_DIR/p7_pool_validation.json" \
    --preload-buffer-pt "$BUF_DIR/p7_opt3a_demo_buf.pt" --pretrain-steps 4500 \
    --rounds 24 --eps-per-round 16 --updates-per-round 400 --max-steps 220 \
    --vla-stride 6 --scale-xyz-mm 2.2 --scale-rot-rad 0.025 --q-weight-max 0.10 \
    --hires-wrist-roi --seed 101 --out-dir "$RLPD_OUT/p7_s101_xyz22_qw10" \
    > "$H2/logs/s2_rlpd_p7_b.log" 2>&1 &
PID_R3=$!

wait "$PID_R0" "$PID_R1" "$PID_R2" "$PID_R3"
echo "=== [STAGE 2 RLPD COMPLETE $(date)] ==="
