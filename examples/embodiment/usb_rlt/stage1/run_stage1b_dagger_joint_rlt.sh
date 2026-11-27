#!/usr/bin/env bash
set -euo pipefail
H2=/vla-cd/jy/v12b_usb_cycle_v1/holder_v2_rlt
EXP12=/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase12_v1
EXP13=/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1
PY=/root/g1_usb_sim2real/.venv-pi05/bin/python
S1A_CKPT="$H2/stage1_pi05_rlt/pi05_v3_fresh_sft_14k/pi05_v3_fresh_sft_14k/checkpoints/global_step_8000/actor/dcp_checkpoint"

GPU_IDS_CSV="${GPU_IDS:-4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS_CSV"
NSHARDS="${#GPU_ARR[@]}"
FIRST_GPU="${GPU_ARR[0]}"

export PYTHONPATH="/root/v12b_usb_cycle_v1/runtime:$H2/code:$EXP12/code:$EXP13/code:$EXP13/dependencies:$EXP13/rlinf:/vla-cd/jy/v12b_usb_cycle_v1:/vla-cd/jy/v12b_usb_cycle_v1/approach_audit_v1:/vla-cd/jy/v12b_usb_cycle_v1/revisions/teacher_axis_return_v1:/root/v12b_usb_cycle_v1/base_code/scripts:/root/v12b_usb_cycle_v1/base_code/src"
export JAX_PLATFORMS=cpu
export TORCH_COMPILE_DISABLE=1
export HF_HOME="$H2/.cache/hf"
export TORCH_HOME="$H2/.cache/torch"
export XDG_CACHE_HOME="$H2/.cache/xdg"

rm -f /mlplatform/jy/v3_fresh_nvme/dagger_s1b/harvested_shard*.json
echo "=== [STAGE 1B-1 ON-POLICY DAGGER ROLLOUT $(date)] GPUS=$GPU_IDS_CSV NSHARDS=$NSHARDS ==="
PIDS=()
for i in $(seq 0 $((NSHARDS - 1))); do
  G="${GPU_ARR[$i]}"
  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G \
    "$PY" -u "$H2/code/collect_v3_onpolicy_dagger.py" rollout \
      --config-dir "$H2/configs" \
      --config-name pi05_v3_fresh_sft_14k \
      --checkpoint "$S1A_CKPT" \
      --limit 400 --max-steps 2600 \
      --shard-idx "$i" --num-shards "$NSHARDS" \
      > "$H2/logs/dagger_s1b_roll_shard${i}.log" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "=== [STAGE 1B-2 DAGGER SMOOTH TEACHER RECOVERY $(date)] ==="
MUJOCO_GL=egl "$PY" -u "$H2/code/collect_v3_onpolicy_dagger.py" recover --workers 48 --egl-gpu-ids "$GPU_IDS_CSV"

echo "=== [STAGE 1B-3 MERGE BASE + DAGGER $(date)] ==="
"$PY" -u "$H2/code/collect_v3_onpolicy_dagger.py" merge --dagger-repeat 2

echo "=== [STAGE 1B-4 JOINT RLT STAGE 1 TRAINING $(date)] GPUS=$GPU_IDS_CSV (mbs=4, gbs=64) ==="
"$PY" -u "$H2/code/run_training_v3_fresh.py" --config-name pi05_v3_fresh_joint_rlt_s1 --gpus "$GPU_IDS_CSV" --micro-batch-size 4

S1B_CKPT="$H2/stage1_pi05_rlt/pi05_v3_fresh_joint_rlt_s1/pi05_v3_fresh_joint_rlt_s1/checkpoints/global_step_2500/actor/dcp_checkpoint"
echo "=== [STAGE 1B-5 JOINT RLT REPRESENTATION PROBE $(date)] ==="
CUDA_VISIBLE_DEVICES=$FIRST_GPU "$PY" -u "$H2/code/probe_rlt_stage1_joint.py" \
  --config-dir "$H2/configs" \
  --config-name pi05_v3_fresh_joint_rlt_s1 \
  --checkpoint "$S1B_CKPT" \
  --dataset-dir /mlplatform/jy/v3_fresh_nvme/train_v3_fresh_bal \
  --num-samples 256 \
  --out-json "$H2/stage1_pi05_rlt/probe_v3_fresh_joint_rlt_s1_s2500.json"

echo "=== [STAGE 1B-6 GATE 1B 34-VAL EVAL $(date)] GPUS=$GPU_IDS_CSV ==="
PIDS=()
for i in $(seq 0 $((NSHARDS - 1))); do
  G="${GPU_ARR[$i]}"
  CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G \
    "$PY" -u "$H2/code/eval_full_cycle_pi05_rlt.py" \
      --label v3fresh_g1b_joint_s2500_val34 \
      --stage1-config-dir "$H2/configs" \
      --stage1-config-name pi05_v3_fresh_joint_rlt_s1 \
      --stage1-ckpt "$S1B_CKPT" \
      --scenes-root "$H2/scenes_v3_real" \
      --split validation --limit 34 --max-steps 3600 \
      --phase-flow-m 4:1 \
      --shard-idx "$i" --num-shards "$NSHARDS" \
      > "$H2/logs/eval_v3fresh_g1b_s2500_shard${i}.log" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done
echo "=== [STAGE 1B COMPLETE $(date)] ==="
flow-m 4:1 \
      --shard-idx "$i" --num-shards "$NSHARDS" \
      > "$H2/logs/eval_v3fresh_g1b_s2500_shard${i}.log" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done
echo "=== [STAGE 1B COMPLETE $(date)] ==="
