# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Collect real end-to-end Stage 1B + Stage 2 P4 -> P5 -> P6 -> P7 handoff snapshots.

Strictly zero test leakage (only train or validation split), pure friction MuJoCo 4.0.
Captures the exact MuJoCo state the instant SafePhase12SwitcherV2 enters P7 after
successful lift -> insert -> withdraw -> transit_to_holder.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

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

from eval_full_cycle_pi05_rlt import load_stage2_actor
from eval_v2_fast import SafePhase12SwitcherV2, infer_chunk, select_v2_rows
from evaluate_closed_loop_phase12 import DEFAULT_PHASE_EXEC_STEPS as P12_EXEC_STEPS
from evaluate_closed_loop_phase13 import parse_phase_int_map
from policy_executor import PolicyExecutor
import rlt_stage2_residual_ac as s2
import robot_contact_utils as rcu
import usb_rel256_phase12 as usb12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "validation"), required=True)
    ap.add_argument("--limit", type=int, default=240)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--stage1-config-dir", default=str(H2 / "configs"))
    ap.add_argument("--stage1-config-name", default="pi05_v3_fresh_joint_rlt_s1")
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--p4-rlt-ckpt", required=True)
    ap.add_argument("--p56-rlt-ckpt", default="")
    ap.add_argument("--scenes-root", default=str(H2 / "scenes_v3_real"))
    ap.add_argument("--scenes-json", default="")
    ap.add_argument("--save-p4-pool", action="store_true", default=False)
    ap.add_argument("--phase-flow-m", default="")
    ap.add_argument("--phase-exec-steps", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pool-prefix", default="p7_real_pool")
    ap.add_argument("--max-steps", type=int, default=2200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    provider = s2.Stage1RLTFeatureProvider(
        Path(args.stage1_config_dir), args.stage1_config_name, args.stage1_ckpt, device
    )
    vla_stream = torch.cuda.Stream(priority=0)
    p4_act, p4_cfg = load_stage2_actor(args.p4_rlt_ckpt, device)
    assert p4_act is not None
    p56_actor = p56_cfg = None
    make_p56_feat = None
    if args.p56_rlt_ckpt:
        from train_stage2_p56_rlt import Stage2P56ResidualActor, make_p56_feat as _make_p56_feat
        make_p56_feat = _make_p56_feat
        ck56 = torch.load(args.p56_rlt_ckpt, map_location="cpu", weights_only=False)
        p56_actor = Stage2P56ResidualActor().to(device).eval()
        p56_actor.load_state_dict(ck56["actor"], strict=True)
        p56_cfg = dict(
            max_dq_p6=float(ck56.get("max_dq_p6", 0.025)),
            max_dq_p10=float(ck56.get("max_dq_p10", 0.028)),
            p5_scale_xyz_m=float(ck56.get("p5_scale_xyz_m", 0.0018)),
            p5_scale_rot_rad=float(ck56.get("p5_scale_rot_rad", 0.022)),
            p5_q_clamp_rad=float(ck56.get("p5_q_clamp_rad", 0.020)),
            arch_version=str(ck56.get("arch_version", "p56_rlt_v1")),
        )

    sr = Path(args.scenes_root)
    val_j = sr / "stage2_validation_scenes_v3.json"
    test_j = sr / "stage2_test_scenes_v3.json"
    val_seeds = {int(s["seed"]) for s in json.loads(val_j.read_text())} if val_j.exists() else set()
    test_seeds = {int(s["seed"]) for s in json.loads(test_j.read_text())} if test_j.exists() else set()

    flow_m_map = parse_phase_int_map(args.phase_flow_m)
    exec_steps_map = parse_phase_int_map(args.phase_exec_steps)
    if args.scenes_json:
        raw_sc = json.loads(Path(args.scenes_json).read_text())
        if args.limit < len(raw_sc):
            idxs = np.linspace(0, len(raw_sc) - 1, min(args.limit, len(raw_sc)), dtype=int)
            raw_sc = [raw_sc[i] for i in idxs]
        rows = [dict(scene=s, record=None) for s in raw_sc]
    else:
        rows = select_v2_rows(args.split, None, args.limit, sr)
    if args.split == "train":
        rows = [r for r in rows if int(r["scene"]["seed"]) not in (val_seeds | test_seeds)]
    else:
        rows = [r for r in rows if int(r["scene"]["seed"]) not in test_seeds]
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_idx]

    tmp_root = Path("/dev/shm/jy_h2_tmp") / f"p7_real_{args.split}_s{args.shard_idx}"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    items = []
    p4_items = []
    t_all = time.time()
    for idx, row in enumerate(rows):
        t0 = time.time()
        sc = row["scene"]
        seed = int(sc["seed"])
        rec_path = Path(row["record"]["path"]) if row.get("record") else None
        run_dir = tmp_root / str(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        env = PolicyExecutor(Path(sc["scene"]), run_dir, seed, brightness=float(sc["brightness"]))
        captured = False
        captured_p4 = False
        try:
            if rec_path is not None and (rec_path / "cycle_phase_raise_to_pregrasp.npz").exists():
                env.reset(snapshot=rec_path / "cycle_phase_raise_to_pregrasp.npz")
            else:
                env.reset()
            task = env._task
            obs0 = env.observe()
            q_home0 = obs0["observation.state"][6:13].astype(np.float32).copy()
            sw = SafePhase12SwitcherV2(start_phase=0, max_grasp_retries=2)
            cur_phase = 0
            pending: list[np.ndarray] = []
            pending_dq: list[np.ndarray] = []
            cached_z = cached_ref = None
            steps_since_vla = 999
            raw0 = obs0["observation.state"][:14].astype(np.float32)
            q_tgt = raw0[6:13].astype(np.float64).copy()
            tcp_entry = raw0[:3].copy()
            q_p2_lift_mem = raw0[6:13].copy()
            last_cmd_q7 = raw0[6:13].copy()
            q_hist = [raw0[6:13].copy()] * 5
            prev_act6 = np.zeros(6, dtype=np.float32)
            stall_steps = 0

            for i in range(args.max_steps):
                obs = env.observe()
                raw_st = obs["observation.state"]
                if i > 0:
                    qf = raw_st[6:13].astype(np.float32)
                    q_hist.append(qf.copy())
                    if len(q_hist) > 5:
                        q_hist.pop(0)
                    stall_steps = (stall_steps + 1) if float(np.linalg.norm(qf - q_hist[0])) < 0.0020 else 0
                fq_margin = np.asarray(task.d.qpos[task.fq] - task.finger_lower, dtype=np.float32)
                st21 = usb12.make_state21(raw_st, q_home0)
                new_phase, cmd_ov = sw.step_update(i, st21, fq_margin, q_home0)
                if new_phase != cur_phase:
                    if cur_phase == 2 and new_phase == 3:
                        q_p2_lift_mem = raw_st[6:13].astype(np.float32).copy()
                    cur_phase = new_phase
                    pending.clear()
                    pending_dq.clear()
                    cached_z = cached_ref = None
                    steps_since_vla = 999
                    q_tgt = raw_st[6:13].astype(np.float64).copy()
                    tcp_entry = raw_st[:3].astype(np.float32).copy()
                    last_cmd_q7 = raw_st[6:13].astype(np.float32).copy()
                    q_hist = [raw_st[6:13].astype(np.float32).copy()] * 5
                    prev_act6 = np.zeros(6, dtype=np.float32)
                    stall_steps = 0

                if args.save_p4_pool and (not captured_p4) and cur_phase == 4:
                    phys4 = env.outcomes()["physical"]
                    if "lift" in phys4.get("event_times", {}):
                        sn4 = f"real_p4_handoff_{args.split}_{seed}"
                        prev_out = task.out
                        task.out = snap_dir
                        task.snapshot(sn4)
                        task.out = prev_out
                        sp4 = snap_dir / f"{sn4}.npz"
                        if not sp4.exists():
                            cands4 = sorted(snap_dir.glob(f"*{sn4}*.npz"))
                            sp4 = cands4[-1] if cands4 else sp4
                        met4 = task.insertion_metrics()
                        p4_items.append(dict(
                            scene=sc, snapshot=str(sp4), q_home0=q_home0.tolist(),
                            source="real_home_handoff", entry_step=i,
                            entry_lat_mm=round(met4["lateral_error_m"] * 1000.0, 3),
                            entry_dep_mm=round(met4["depth_m"] * 1000.0, 3),
                        ))
                        captured_p4 = True

                if cur_phase == 7:
                    phys = env.outcomes()["physical"]
                    evs = phys.get("event_times", {})
                    if "lift" in evs and "insert" in evs and "withdraw" in evs and float(np.sum(fq_margin)) >= 0.00105:
                        snap_name = f"real_p7_handoff_{args.split}_{seed}"
                        prev_out = task.out
                        task.out = snap_dir
                        task.snapshot(snap_name)
                        task.out = prev_out
                        saved_npz = snap_dir / f"{snap_name}.npz"
                        if not saved_npz.exists():
                            cands = sorted(snap_dir.glob(f"*{snap_name}*.npz"))
                            saved_npz = cands[-1] if cands else saved_npz
                        assert saved_npz.exists() and saved_npz.with_suffix(".controller.json").exists(), (
                            f"Missing snapshot files for {snap_name} in {snap_dir}"
                        )
                        tip, Rp = task.object_pose()
                        delta = tip - task.initial_tip
                        ang = float(np.degrees(np.linalg.norm(Rotation.from_matrix(task.initial_rotation @ Rp.T).as_rotvec())))
                        items.append(dict(
                            scene=sc,
                            snapshot=str(saved_npz),
                            q_home0=q_home0.tolist(),
                            source="real_home_handoff",
                            handoff_mode="p56_v2" if p56_actor is not None else "stage1b",
                            entry_step=i,
                            entry_hxy_mm=round(float(np.linalg.norm(delta[:2]) * 1000.0), 3),
                            entry_hdz_mm=round(float(abs(delta[2]) * 1000.0), 3),
                            entry_ang_deg=round(ang, 3),
                        ))
                        captured = True
                    break

                if cmd_ov is not None:
                    cmd = cmd_ov
                    pending.clear()
                    pending_dq.clear()
                elif p56_actor is not None and cur_phase in (5, 6):
                    if cur_phase == 5:
                        if not pending_dq:
                            if cached_z is None or steps_since_vla >= 12:
                                cached_z, _, _ = provider.infer(obs, q_home0, 5, step_seed=7317 + i, flow_m=1)
                                steps_since_vla = 0
                            f56 = make_p56_feat(5, raw_st[6:13], q_p2_lift_mem, raw_st[:3], tcp_entry)
                            with torch.no_grad():
                                a7 = p56_actor(
                                    torch.from_numpy(cached_z[None]).to(device),
                                    torch.from_numpy(f56[None]).to(device),
                                )[0].cpu().numpy().astype(np.float32)
                            dq = s2.step_persistent_qtgt_dq(
                                task, a7[:6],
                                scale_xyz_m=p56_cfg["p5_scale_xyz_m"],
                                scale_rot_rad=p56_cfg["p5_scale_rot_rad"],
                                max_dq_rad=0.025,
                                exact_nullspace=True,
                            )
                            pending_dq = [0.5 * dq, 0.5 * dq]
                        qc = raw_st[6:13].astype(np.float64)
                        q_tgt = qc + np.clip((q_tgt + pending_dq.pop(0)) - qc, -p56_cfg["p5_q_clamp_rad"], p56_cfg["p5_q_clamp_rad"])
                        cmd = np.zeros(8, dtype=np.float32)
                        cmd[:7] = q_tgt.astype(np.float32)
                        cmd[7] = 0.98
                        if float(np.min(fq_margin)) < 0.00018 and float(np.sum(fq_margin)) < 0.00085:
                            cmd[7] = 0.0
                        steps_since_vla += 1
                    else:
                        if cached_z is None or steps_since_vla >= 12:
                            cached_z, _, _ = provider.infer(obs, q_home0, 6, step_seed=7317 + i, flow_m=1)
                            steps_since_vla = 0
                        max_dq_use = p56_cfg["max_dq_p6"]
                        f56 = make_p56_feat(6, raw_st[6:13], q_p2_lift_mem, raw_st[:3], tcp_entry, max_dq_joint=max_dq_use)
                        with torch.no_grad():
                            a7 = p56_actor(
                                torch.from_numpy(cached_z[None]).to(device),
                                torch.from_numpy(f56[None]).to(device),
                            )[0].cpu().numpy().astype(np.float32)
                        qc = raw_st[6:13].astype(np.float64)
                        dq_step = np.clip((a7.astype(np.float64) / 0.96) * max_dq_use, -max_dq_use, max_dq_use)
                        cmd = np.zeros(8, dtype=np.float32)
                        cmd[:7] = (qc + dq_step).astype(np.float32)
                        cmd[7] = 0.98
                        if float(np.min(fq_margin)) < 0.00018 and float(np.sum(fq_margin)) < 0.00085:
                            cmd[7] = 0.0
                        steps_since_vla += 1
                elif cur_phase == 4:
                    if not pending_dq:
                        if cached_z is None or steps_since_vla >= p4_cfg["vla_stride"]:
                            cached_z, _, cached_ref = provider.infer(obs, q_home0, 4, step_seed=7317 + i, flow_m=1)
                            steps_since_vla = 0
                        wc = s2.extract_wrist_128(task, obs, hires_wrist=p4_cfg["hires_wrist_roi"])
                        fc = s2.extract_front_96(task, obs, "p4_insert", hires_front=p4_cfg["hires_front_roi"])
                        lc = s2.extract_left_96(task, obs, "p4_insert", hires_left=p4_cfg["hires_left_roi"])
                        pf = rcu.extract_contact_proprio(task, raw_st, last_cmd_q7, q_hist[0], tcp_entry, stall_steps, prev_act6)
                        with torch.no_grad():
                            lt = torch.from_numpy(lc[None]).to(device) if lc is not None else None
                            act6 = p4_act(
                                torch.from_numpy(cached_z[None]).to(device),
                                torch.from_numpy(pf[None]).to(device),
                                torch.from_numpy(cached_ref[None]).to(device),
                                torch.from_numpy(wc[None]).to(device),
                                torch.from_numpy(fc[None]).to(device),
                                lt,
                            )[0].cpu().numpy().astype(np.float32)
                        dq = s2.step_persistent_qtgt_dq(
                            task, act6,
                            scale_xyz_m=p4_cfg["scale_xyz_m"],
                            scale_rot_rad=p4_cfg["scale_rot_rad"],
                            max_dq_rad=0.025,
                            exact_nullspace=p4_cfg["exact_nullspace"],
                        )
                        prev_act6 = act6.copy()
                        pending_dq = [0.5 * dq, 0.5 * dq]
                    qc = raw_st[6:13].astype(np.float64)
                    q_tgt = qc + np.clip((q_tgt + pending_dq.pop(0)) - qc, -p4_cfg["q_clamp_rad"], p4_cfg["q_clamp_rad"])
                    cmd = np.zeros(8, dtype=np.float32)
                    cmd[:7] = q_tgt.astype(np.float32)
                    cmd[7] = 0.98
                    if float(np.min(fq_margin)) < 0.00018 and float(np.sum(fq_margin)) < 0.00085:
                        cmd[7] = 0.0
                    steps_since_vla += 1
                else:
                    if not pending:
                        raw_in = dict(
                            state=st21,
                            image=obs["observation.images.front"],
                            left_wrist_image=obs["observation.images.left_wrist"],
                            wrist_image=obs["observation.images.wrist"],
                            prompt=usb12.PHASE_PROMPTS[cur_phase],
                        )
                        fm = int(flow_m_map.get(cur_phase, 1))
                        pred = infer_chunk(provider.vla, provider.tx, provider.un, raw_in, i, flow_m=fm, stream=vla_stream)
                        default_k = min(4, P12_EXEC_STEPS[cur_phase]) if cur_phase <= 3 else P12_EXEC_STEPS[cur_phase]
                        k_exec = min(int(exec_steps_map.get(cur_phase, default_k)), len(pred))
                        pending = list(pred[:k_exec])
                    cmd = np.asarray(pending.pop(0), dtype=np.float32).copy()
                    if cmd[7] >= 0.85 and ((float(np.min(fq_margin)) < 0.00018 and float(np.sum(fq_margin)) < 0.00085) or float(np.min(fq_margin)) < 0.00008):
                        cmd[7] = 0.0

                res = env.step(cmd)
                last_cmd_q7 = cmd[:7].copy()
                if res["terminated"] or res["truncated"]:
                    break

            print(
                f"[{args.split} s={args.shard_idx} {idx+1}/{len(rows)}] seed={seed} captured_p7={captured} "
                f"final_p={cur_phase} elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
        finally:
            s2.close_hires_wrist_renderer(env._task)
            env.close()
            shutil.rmtree(run_dir, ignore_errors=True)

    shutil.rmtree(tmp_root, ignore_errors=True)
    sfx = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""
    out_fp = out_dir / f"{args.pool_prefix}_{args.split}{sfx}.json"
    out_fp.write_text(json.dumps(items, indent=2) + "\n")
    if args.save_p4_pool:
        p4_fp = out_dir / f"p4_real_pool_{args.split}{sfx}.json"
        p4_fp.write_text(json.dumps(p4_items, indent=2) + "\n")
    print(f"DONE {args.split}{sfx}: captured p7={len(items)}/{len(rows)} p4={len(p4_items)}/{len(rows)} in {time.time()-t_all:.1f}s -> {out_fp}", flush=True)


if __name__ == "__main__":
    main()
