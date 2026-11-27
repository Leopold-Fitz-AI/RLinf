# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Collect Real Upstream pi0.5 Handoff Snapshot Pools (Pool A: Home->P4, Pool B: P7 holder) on scenes_v2_keepvis.

Strictly separates train vs validation seeds (never touches test seeds).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/vla-cd/jy/v12b_usb_cycle_v1")
EXP12 = ROOT / "clean_rel256_phase12_v1"
EXP13 = ROOT / "clean_rel256_phase13_v1"
H2 = ROOT / "holder_v2_rlt"

for p in [
    "/root/v12b_usb_cycle_v1/runtime",
    str(H2 / "code"),
    str(EXP12 / "code"),
    str(EXP13 / "code"),
    str(EXP13 / "dependencies"),
    str(EXP13 / "rlinf"),
    str(ROOT),
    str(ROOT / "approach_audit_v1"),
    str(ROOT / "revisions/teacher_axis_return_v1"),
    "/root/v12b_usb_cycle_v1/base_code/scripts",
    "/root/v12b_usb_cycle_v1/base_code/src",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from policy_executor import PolicyExecutor
from eval_v2_fast import SafePhase12SwitcherV2, infer_chunk, load_model_fp32, select_v2_rows
from evaluate_closed_loop_phase12 import DEFAULT_PHASE_EXEC_STEPS as P12_EXEC_STEPS
import usb_rel256_phase12 as usb12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(H2 / "configs"))
    ap.add_argument("--config-name", default="usb_unified_v2_rlt_stage1")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=("train", "validation"), required=True)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--scenes-root", default=str(H2 / "scenes_v3_real"))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    usb12.register()
    out_dir = Path(args.out_dir)
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    sr = Path(args.scenes_root)
    raw_v3_root = Path("/mlplatform/jy/v3_fresh_nvme/raw_full_1200")
    v3_1200_scenes = Path("/mlplatform/jy/v3_fresh_nvme/train_scenes_v3_1200.json")
    if args.split == "train" and "v3" in sr.name and raw_v3_root.exists() and v3_1200_scenes.exists():
        val_seeds = {int(s["seed"]) for s in json.loads((sr / "stage2_validation_scenes_v3.json").read_text())}
        test_seeds = {int(s["seed"]) for s in json.loads((sr / "stage2_test_scenes_v3.json").read_text())}
        forbid = val_seeds | test_seeds
        all_rows = []
        for sc in json.loads(v3_1200_scenes.read_text()):
            sd = int(sc["seed"])
            assert sd not in forbid, f"LEAKAGE seed={sd}"
            v3_ep = raw_v3_root / str(sd)
            if (v3_ep / "v3f_done.json").exists() and (v3_ep / "demonstration_full.npz").exists():
                all_rows.append(dict(scene=sc, record=dict(path=str(v3_ep), parent_scene_seed=sd)))
        if args.limit < len(all_rows):
            idxs = np.linspace(0, len(all_rows) - 1, args.limit, dtype=int)
            rows = [all_rows[i] for i in idxs]
        else:
            rows = all_rows
    else:
        rows = select_v2_rows(args.split, None, args.limit, sr)
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_idx]

    model, tx, un = load_model_fp32(Path(args.config_dir), args.config_name, args.checkpoint)
    stream = torch.cuda.Stream(priority=0)

    p4_pool = []
    p7_pool = []
    t0 = time.time()

    for idx, row in enumerate(rows):
        sc = row["scene"]
        seed = int(sc["seed"])
        rec_path = Path(row["record"]["path"])

        # 1) P7 holder pool: always include teacher replace_in_holder arrival snapshot on v2_keepvis
        with np.load(rec_path / "demonstration_full.npz", allow_pickle=True) as d:
            q_home0_demo = d["observation.state"][0, 6:13].astype(np.float32).tolist()
        p7_snap = rec_path / "cycle_phase_replace_in_holder.npz"
        if p7_snap.exists():
            p7_pool.append(dict(scene=sc, snapshot=str(p7_snap), q_home0=q_home0_demo, source="replace_in_holder"))

        # 2) P4 real upstream handoff pool: run Stage 1 pi0.5 from Home (P0 -> P3 -> P4 entry!)
        run_dir = Path("/dev/shm/jy_h2_tmp") / f"pool_{args.split}_{seed}_{args.shard_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        env = PolicyExecutor(Path(sc["scene"]), run_dir, seed, brightness=float(sc["brightness"]))
        try:
            env.reset(snapshot=rec_path / "cycle_phase_raise_to_pregrasp.npz")
            task = env._task
            obs0 = env.observe()
            q_home0 = np.asarray(obs0["observation.state"][6:13], dtype=np.float32).copy()
            sw = SafePhase12SwitcherV2(start_phase=0, max_grasp_retries=2)
            pending = []
            cur_phase = 0
            captured_p4 = False

            for i in range(1600):
                obs = env.observe()
                raw_st = obs["observation.state"]
                fq_margin = np.asarray(task.d.qpos[task.fq] - task.finger_lower, dtype=np.float32)
                st21 = usb12.make_state21(raw_st, q_home0)
                new_phase, cmd_ov = sw.step_update(i, st21, fq_margin, q_home0)
                if new_phase != cur_phase:
                    cur_phase = new_phase
                    pending.clear()

                # Capture exact MuJoCo snapshot the instant pi0.5 enters P4 with USB held!
                if cur_phase == 4:
                    phys = env.outcomes()["physical"]
                    if "lift" in phys.get("event_times", {}):
                        # Save exact task snapshot directly compatible with PolicyExecutor.reset(snapshot=...)
                        snap_name = f"real_p4_handoff_{args.split}_{seed}"
                        task.out = snap_dir
                        task.snapshot(snap_name)
                        saved_npz = snap_dir / f"cycle_phase_{snap_name}.npz"
                        if not saved_npz.exists():
                            # Find whichever file task.snapshot wrote
                            cands = sorted(snap_dir.glob(f"*{snap_name}*.npz"))
                            saved_npz = cands[-1] if cands else snap_dir / f"{snap_name}.npz"
                        met = task.insertion_metrics()
                        p4_pool.append(dict(
                            scene=sc,
                            snapshot=str(saved_npz),
                            q_home0=q_home0.tolist(),
                            source="real_home_handoff",
                            entry_step=i,
                            entry_lat_mm=round(met["lateral_error_m"] * 1000.0, 3),
                            entry_dep_mm=round(met["depth_m"] * 1000.0, 3),
                            entry_ang_deg=round(float(np.degrees(met["angle_error_rad"])), 3),
                        ))
                        captured_p4 = True
                        break

                if cmd_ov is not None:
                    cmd = cmd_ov
                    pending.clear()
                else:
                    if not pending:
                        raw_in = dict(
                            state=st21,
                            image=obs["observation.images.front"],
                            left_wrist_image=obs["observation.images.left_wrist"],
                            wrist_image=obs["observation.images.wrist"],
                            prompt=usb12.PHASE_PROMPTS[cur_phase],
                        )
                        pred = infer_chunk(model, tx, un, raw_in, i, flow_m=1, stream=stream)
                        pending = list(pred[: min(4, P12_EXEC_STEPS[cur_phase])])
                    cmd = np.asarray(pending.pop(0), dtype=np.float32).copy()
                    if cmd[7] >= 0.85 and float(np.min(fq_margin)) < 0.00018 and float(np.sum(fq_margin)) < 0.00085:
                        cmd[7] = 0.0
                res = env.step(cmd)
                if res["terminated"] or res["truncated"]:
                    break

            print(
                f"[{args.split} shard={args.shard_idx} {idx+1}/{len(rows)}] seed={seed} captured_p4={captured_p4} "
                f"final_p={cur_phase} elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
        finally:
            env.close()
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)

    # Also add clean align_at_mouth snapshots for train curriculum (optional, tagged separately)
    if args.split == "train":
        for row in rows:
            sc = row["scene"]
            rec_path = Path(row["record"]["path"])
            m_snap = rec_path / "cycle_phase_align_at_mouth.npz"
            if m_snap.exists():
                with np.load(rec_path / "demonstration_full.npz", allow_pickle=True) as d:
                    qh = d["observation.state"][0, 6:13].astype(np.float32).tolist()
                p4_pool.append(dict(scene=sc, snapshot=str(m_snap), q_home0=qh, source="mouth_nominal"))

    sfx = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""
    (out_dir / f"p4_pool_{args.split}{sfx}.json").write_text(json.dumps(p4_pool, indent=2) + "\n")
    (out_dir / f"p7_pool_{args.split}{sfx}.json").write_text(json.dumps(p7_pool, indent=2) + "\n")
    print(f"DONE {args.split}{sfx}: p4_pool={len(p4_pool)}, p7_pool={len(p7_pool)}", flush=True)


if __name__ == "__main__":
    main()
