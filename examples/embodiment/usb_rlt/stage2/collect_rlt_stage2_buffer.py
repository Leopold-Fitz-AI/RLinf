# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fast Parallel RLPD Demo Buffer Collector (12x parallel EGL sim + Batched GPU pi0.5+RLT encoding).

Collects diverse pure-friction DART (noise-injected recovery) + clean expert trajectories across ALL
372 unique training scenes + real Home->P4 handoffs (STRICTLY disjoint from val/test seeds!), then
encodes (z_rl, ref_rel32) in GPU batches of 16 via frozen Stage 1 pi0.5 + RLTTokenTransformer.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

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
from rlinf.models.embodiment.openpi.modules import model as openpi_model
import rlt_stage2_residual_ac as r2
import robot_contact_utils as rcu
import usb_rel256_phase12 as usb12


def _worker_collect(job_args: tuple) -> dict:
    (
        idx, item, subtask, use_perturb, dart_sigma, max_steps,
        scale_xyz_m, scale_rot_rad, q_clamp_rad, vla_stride, seed_off, tmp_root_str,
        hires_wrist, hires_front, hires_left, exact_nullspace, noback_p4, soft_p7, p7_q2lift,
        smooth_p7, p7_inhand_rand, bright_rand,
    ) = job_args
    os.environ["MUJOCO_GL"] = "egl"
    egl_list = [x.strip() for x in os.environ.get("STAGE2_EGL_GPUS", "").split(",") if x.strip()]
    if egl_list:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = egl_list[idx % len(egl_list)]
    else:
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(idx % 4))
    rng = np.random.default_rng(10000 + seed_off + idx * 97)
    sc = item["scene"]
    seed = int(sc["seed"])
    q_home0 = np.asarray(item["q_home0"], dtype=np.float32)
    run_dir = Path(tmp_root_str) / f"w_{os.getpid()}_{idx}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    br_use = float(rng.uniform(0.76, 1.24)) if bright_rand else float(sc["brightness"])
    env = PolicyExecutor(Path(sc["scene"]), run_dir, seed, brightness=br_use)
    target_event = "insert" if subtask == "p4_insert" else "return"
    try:
        env.reset(snapshot=Path(item["snapshot"]))
        task = env._task
        obs = env.observe()
        if use_perturb:
            ok_p = r2.apply_train_only_initial_perturbation(
                env, task, subtask, rng, soft_p7=soft_p7, p7_q2lift=p7_q2lift, p7_inhand_rand=p7_inhand_rand,
            )
            if not ok_p:
                env.reset(snapshot=Path(item["snapshot"]))
            obs = env.observe()

        raw14 = obs["observation.state"][:14].astype(np.float32)
        tcp_entry = raw14[:3].copy()
        q_tgt = raw14[6:13].astype(np.float64).copy()
        last_cmd_q7 = raw14[6:13].copy()
        q_hist = [raw14[6:13].copy()] * 5
        prev_act6 = np.zeros(6, dtype=np.float32)
        stall_steps = 0
        pot_prev, _ = r2.compute_subtask_potential(task, subtask)

        # Store raw observations at vla_stride boundaries so GPU can batch-encode (z_rl, ref_rel32)
        vla_frames: list[dict] = []
        trs: list[dict] = []
        steps_since_vla = vla_stride
        cur_vla_idx = -1
        success = False
        safety_stop = False
        step_done = 0
        p7_hold_steps = 0
        use_geom_return = bool(
            p7_q2lift and subtask == "p7_holder"
            and ("withdraw" not in env.outcomes()["physical"].get("event_times", {}))
        )

        for step_i in range(0, max_steps, 2):
            if cur_vla_idx < 0 or steps_since_vla >= vla_stride:
                cur_vla_idx = len(vla_frames)
                vla_frames.append(dict(
                    state21=usb12.make_state21(obs["observation.state"], q_home0).astype(np.float32),
                    front=obs["observation.images.front"].copy(),
                    left_wrist=obs["observation.images.left_wrist"].copy(),
                    wrist=obs["observation.images.wrist"].copy(),
                ))
                steps_since_vla = 0

            w_crop = r2.extract_wrist_128(task, obs, hires_wrist=hires_wrist)
            f_crop = r2.extract_front_96(task, obs, subtask, hires_front=hires_front)
            l_crop = r2.extract_left_96(task, obs, subtask, hires_left=hires_left)
            p_feat = rcu.extract_contact_proprio(
                task, obs["observation.state"], last_cmd_q7, q_hist[0], tcp_entry, stall_steps, prev_act6
            )
            g_feat = r2.compute_train_only_geom6(task, subtask)

            # Expert action label (always clean recovery action for THIS exact state!)
            exp_act6 = r2.compute_train_only_expert_act6(
                task, subtask, scale_xyz_m=scale_xyz_m, scale_rot_rad=scale_rot_rad,
                compliant_p4=exact_nullspace, noback_p4=noback_p4, soft_p7=soft_p7, smooth_p7=smooth_p7,
            )
            # DART: inject mild noise into executed action so expert demonstrates rim/offset recovery!
            if dart_sigma > 0.0 and step_i < max_steps - 60:
                exec_act6 = np.clip(exp_act6 + rng.normal(0.0, dart_sigma, size=6).astype(np.float32), -1.0, 1.0)
            else:
                exec_act6 = exp_act6

            dq = r2.step_persistent_qtgt_dq(
                task, exec_act6, scale_xyz_m=scale_xyz_m, scale_rot_rad=scale_rot_rad, max_dq_rad=0.025, exact_nullspace=exact_nullspace
            )
            terminated_now = False
            for _ in range(2):
                q_cur = obs["observation.state"][6:13].astype(np.float64)
                q_tgt = q_cur + np.clip((q_tgt + 0.5 * dq) - q_cur, -q_clamp_rad, q_clamp_rad)
                cmd = np.zeros(8, dtype=np.float32)
                cmd[:7] = q_tgt.astype(np.float32)
                cmd[7] = 0.98
                fq_margin = np.asarray(task.d.qpos[task.fq] - task.finger_lower, dtype=np.float32)
                if float(np.min(fq_margin)) < 0.00018 and float(np.sum(fq_margin)) < 0.00085:
                    cmd[7] = 0.0
                res = env.step(cmd)
                last_cmd_q7 = cmd[:7].copy()
                step_done += 1
                steps_since_vla += 1
                obs = env.observe()
                q_cur2 = obs["observation.state"][6:13].astype(np.float32)
                q_hist.append(q_cur2.copy())
                if len(q_hist) > 5:
                    q_hist.pop(0)
                stall_steps = (stall_steps + 1) if float(np.linalg.norm(q_cur2 - q_hist[0])) < 0.0020 else 0
                phys = env.outcomes()["physical"]
                if use_geom_return:
                    tip_h, _ = task.object_pose()
                    hxy_h = float(np.linalg.norm((tip_h - task.initial_tip)[:2])) * 1000.0
                    dz_h = abs(float(tip_h[2] - task.initial_tip[2])) * 1000.0
                    fq_s = float(np.sum(task.d.qpos[task.fq] - task.finger_lower))
                    if hxy_h < 2.0 and dz_h < 2.2 and fq_s >= 0.00105:
                        p7_hold_steps += 1
                        if p7_hold_steps >= 18:
                            success = True
                            terminated_now = True
                            break
                    else:
                        p7_hold_steps = 0
                elif target_event in phys.get("event_times", {}):
                    if soft_p7 and subtask == "p7_holder":
                        tip_h, _ = task.object_pose()
                        dz_h = abs(float(tip_h[2] - task.initial_tip[2])) * 1000.0
                        fq_s = float(np.sum(task.d.qpos[task.fq] - task.finger_lower))
                        if dz_h < 2.2 and fq_s >= 0.00105:
                            p7_hold_steps += 1
                            if p7_hold_steps >= 18:
                                success = True
                                terminated_now = True
                                break
                        else:
                            p7_hold_steps = 0
                    else:
                        success = True
                        terminated_now = True
                        break
                if res["terminated"] or res["truncated"]:
                    safety_stop = (phys.get("invalid_reason") == "safety_stop")
                    terminated_now = True
                    break

            pot_next, info_next = r2.compute_subtask_potential(task, subtask)
            r_step = float(pot_next - pot_prev) - 0.005 * float(np.mean(exp_act6 ** 2)) - (0.04 if stall_steps >= 10 else 0.0)
            if subtask == "p4_insert":
                r_step += 0.08 * float(np.clip(info_next["dep_mm"], 0.0, 11.6)) / 11.6
            else:
                r_step += 0.08 * float(np.clip(18.0 - info_next["hdz_mm"], 0.0, 18.0)) / 18.0
                fq_min_now = float(np.min(task.d.qpos[task.fq] - task.finger_lower))
                if fq_min_now < 0.00020:
                    r_step -= 2.0
                elif fq_min_now < 0.00035:
                    r_step -= 0.4
            if success:
                r_step += 15.0
            elif safety_stop:
                r_step -= 10.0

            done_flag = 1.0 if (success or safety_stop) else 0.0
            prev_vla_idx = cur_vla_idx
            if not terminated_now and steps_since_vla >= vla_stride:
                cur_vla_idx = len(vla_frames)
                vla_frames.append(dict(
                    state21=usb12.make_state21(obs["observation.state"], q_home0).astype(np.float32),
                    front=obs["observation.images.front"].copy(),
                    left_wrist=obs["observation.images.left_wrist"].copy(),
                    wrist=obs["observation.images.wrist"].copy(),
                ))
                steps_since_vla = 0

            next_w = r2.extract_wrist_128(task, obs, hires_wrist=hires_wrist)
            next_f = r2.extract_front_96(task, obs, subtask, hires_front=hires_front)
            next_l = r2.extract_left_96(task, obs, subtask, hires_left=hires_left)
            next_p = rcu.extract_contact_proprio(
                task, obs["observation.state"], last_cmd_q7, q_hist[0], tcp_entry, stall_steps, exec_act6
            )
            next_g = r2.compute_train_only_geom6(task, subtask)

            # Label with clean exp_act6 (DART supervision!) while recording actual transitions
            tr_rec = dict(
                vla_idx=prev_vla_idx, nvla_idx=cur_vla_idx,
                p=p_feat, w=w_crop, f=f_crop, g=g_feat,
                a=exp_act6.astype(np.float32), a_exec=exec_act6.astype(np.float32),
                r=float(r_step), np=next_p, nw=next_w, nf=next_f, ng=next_g,
                done=float(done_flag), ep_succ=False, is_expert=True, mc=0.0,
            )
            if hires_left:
                tr_rec["l"] = l_crop
                tr_rec["nl"] = next_l
            trs.append(tr_rec)
            prev_act6 = exec_act6.copy()
            pot_prev = pot_next
            if terminated_now:
                break

        G = 0.0
        for tr in reversed(trs):
            G = tr["r"] + 0.98 * G * (1.0 - tr["done"])
            tr["mc"] = float(G)
            tr["ep_succ"] = bool(success)
            tr["is_expert"] = bool(success)

        return dict(seed=seed, success=success, safety=safety_stop, steps=step_done, vla_frames=vla_frames, trs=trs)
    finally:
        r2.close_hires_wrist_renderer(env._task)
        env.close()
        shutil.rmtree(run_dir, ignore_errors=True)


@torch.inference_mode()
def batch_encode_vla_frames(
    provider: r2.Stage1RLTFeatureProvider,
    frames: list[dict],
    phase_idx: int,
    *,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched GPU encoding of (z_rl_fp16, ref_rel32_fp32) — 10-15x faster than batch=1!"""
    N = len(frames)
    z_out = np.empty((N, 2048), dtype=np.float16)
    ref_out = np.empty((N, 32), dtype=np.float32)
    prompt = usb12.PHASE_PROMPTS[phase_idx]
    for s in range(0, N, batch_size):
        chunk = frames[s : s + batch_size]
        procs = []
        for fr in chunk:
            raw = dict(
                state=fr["state21"],
                image=fr["front"],
                left_wrist_image=fr["left_wrist"],
                wrist_image=fr["wrist"],
                prompt=prompt,
            )
            p = provider.tx(raw)
            procs.append({k: v for k, v in p.items() if k != "actions"})
        batch = provider.vla._observation_to_device(default_collate(procs))
        B = len(chunk)
        torch.manual_seed(7317 + s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            obs_prep = openpi_model.preprocess_observation(batch, train=False)
            prefix_out, prefix_mask, kv_cache = provider.vla.build_prefix_cache(obs_prep)
            z_b = provider.rlt.encode_flat(prefix_out.float(), prefix_mask).float().cpu().numpy()
            x_t = torch.randn((B, 16, 32), device=provider.device, dtype=torch.float32)
            dt = -1.0 / 5.0
            t_val = 1.0
            while t_val >= -dt / 2:
                t_tensor = torch.full((B,), t_val, device=provider.device, dtype=torch.float32)
                sout = provider.vla.run_suffix(obs_prep, x_t, t_tensor, kv_cache, prefix_mask)
                vt = provider.vla.velocity_from_suffix(sout)
                x_t = x_t + dt * vt.float()
                t_val += dt
            pred_norm = x_t.float().cpu().numpy()
        z_out[s : s + B] = z_b.astype(np.float16)
        ref_out[s : s + B] = pred_norm[:, :4, :8].reshape(B, 32).astype(np.float32)
        if (s // batch_size) % 25 == 0:
            print(f"  [GPU encode] {min(s + B, N)}/{N}", flush=True)
    return z_out, ref_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", choices=("p4_insert", "p7_holder"), required=True)
    ap.add_argument("--stage1-config-dir", default=str(EXP12 / "configs"))
    ap.add_argument("--stage1-config-name", default="phase12_sft8")
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--rlt-adapter-pt", default="")
    ap.add_argument("--scenes-root", default=str(H2 / "scenes_v3_real"))
    ap.add_argument("--train-pool-json", required=True)
    ap.add_argument("--val-pool-json", required=True)
    ap.add_argument("--out-pt", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--vla-stride", type=int, default=6)
    ap.add_argument("--scale-xyz-mm", type=float, default=1.8)
    ap.add_argument("--scale-rot-rad", type=float, default=0.022)
    ap.add_argument("--q-clamp-rad", type=float, default=0.028)
    ap.add_argument("--max-all-scenes", type=int, default=500)
    ap.add_argument("--hires-wrist-roi", action="store_true", default=False)
    ap.add_argument("--hires-front-roi", action="store_true", default=False)
    ap.add_argument("--hires-left-roi", action="store_true", default=False)
    ap.add_argument("--exact-nullspace", action="store_true", default=False)
    ap.add_argument("--noback-p4", action="store_true", default=False)
    ap.add_argument("--soft-p7", action="store_true", default=False)
    ap.add_argument("--smooth-p7", action="store_true", default=False)
    ap.add_argument("--p7-q2lift", action="store_true", default=False)
    ap.add_argument("--p7-inhand-rand", "--inhand-rand", dest="p7_inhand_rand", action="store_true", default=False)
    ap.add_argument("--bright-rand", action="store_true", default=False)
    args = ap.parse_args()

    t0 = time.time()
    val_pool = json.loads(Path(args.val_pool_json).read_text())
    val_seeds = {int(x["scene"]["seed"]) for x in val_pool}
    sr = Path(args.scenes_root)
    test_j = sr / ("stage2_test_scenes_v3.json" if (sr / "stage2_test_scenes_v3.json").exists() else "stage2_test_scenes_v2.json")
    test_seeds = {int(s["seed"]) for s in json.loads(test_j.read_text())}
    val100_j = sr / "val100_unseen_scenes_v3.json"
    val100_seeds = {int(s["seed"]) for s in json.loads(val100_j.read_text())} if val100_j.exists() else set()
    val34_j = sr / "stage2_validation_scenes_v3.json"
    val34_seeds = {int(s["seed"]) for s in json.loads(val34_j.read_text())} if val34_j.exists() else set()
    forbid = val_seeds | val34_seeds | val100_seeds | test_seeds

    train_pool = json.loads(Path(args.train_pool_json).read_text())
    real_tr = [x for x in train_pool if x.get("source") == "real_home_handoff" and int(x["scene"]["seed"]) not in forbid]
    all372 = r2.build_all_train_perturbed_pool(args.subtask, sr, forbid, p7_q2lift=bool(args.p7_q2lift))
    if args.max_all_scenes > 0 and len(all372) > args.max_all_scenes:
        idxs = np.linspace(0, len(all372) - 1, args.max_all_scenes, dtype=int)
        all372 = [all372[i] for i in idxs]
    assert {int(x["scene"]["seed"]) for x in real_tr}.isdisjoint(forbid)
    assert {int(x["scene"]["seed"]) for x in all372}.isdisjoint(forbid)

    tmp_root = Path("/dev/shm/jy_h2_tmp") / f"collect_{args.subtask}_{os.getpid()}"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    hw = bool(args.hires_wrist_roi)
    hf = bool(args.hires_front_roi)
    hl = bool(args.hires_left_roi)
    en = bool(args.exact_nullspace)
    nb = bool(args.noback_p4)
    sp7 = bool(args.soft_p7)
    sm7 = bool(args.smooth_p7)
    q2l = bool(args.p7_q2lift)
    ihr = bool(args.p7_inhand_rand)
    br_r = bool(args.bright_rand)
    jobs = []
    if args.subtask == "p4_insert":
        # 2 passes over all real Home->P4 handoffs (clean + DART noise recovery)
        for it in real_tr:
            jobs.append((len(jobs), it, args.subtask, False, 0.0, args.max_steps, args.scale_xyz_mm * 1e-3, args.scale_rot_rad, args.q_clamp_rad, args.vla_stride, 100, str(tmp_root), hw, hf, hl, en, nb, sp7, q2l, sm7, ihr, br_r))
            jobs.append((len(jobs), it, args.subtask, False, 0.16, args.max_steps, args.scale_xyz_mm * 1e-3, args.scale_rot_rad, args.q_clamp_rad, args.vla_stride, 200, str(tmp_root), hw, hf, hl, en, nb, sp7, q2l, sm7, ihr, br_r))
        # 1 pass over unique train scenes with physical perturbation (60% DART noise recovery)
        for j, it in enumerate(all372):
            dart = 0.16 if (j % 5 < 3) else 0.0
            jobs.append((len(jobs), it, args.subtask, True, dart, args.max_steps, args.scale_xyz_mm * 1e-3, args.scale_rot_rad, args.q_clamp_rad, args.vla_stride, 300, str(tmp_root), hw, hf, hl, en, nb, sp7, q2l, sm7, ihr, br_r))
    else:
        # P7 holder: real P7 handoffs (2 passes) + unique train scenes (90% perturbed w/ in-hand slip+tilt + 60% DART noise)
        for it in real_tr:
            jobs.append((len(jobs), it, args.subtask, False, 0.0, args.max_steps, args.scale_xyz_mm * 1e-3, args.scale_rot_rad, args.q_clamp_rad, args.vla_stride, 350, str(tmp_root), hw, hf, hl, en, nb, sp7, q2l, sm7, ihr, br_r))
            jobs.append((len(jobs), it, args.subtask, False, 0.14, args.max_steps, args.scale_xyz_mm * 1e-3, args.scale_rot_rad, args.q_clamp_rad, args.vla_stride, 375, str(tmp_root), hw, hf, hl, en, nb, sp7, q2l, sm7, ihr, br_r))
        for j, it in enumerate(all372):
            use_p = (j % 10 < (9 if (sp7 or sm7 or ihr) else 7))
            dart = 0.14 if (j % 5 < 3) else 0.0
            jobs.append((len(jobs), it, args.subtask, use_p, dart, args.max_steps, args.scale_xyz_mm * 1e-3, args.scale_rot_rad, args.q_clamp_rad, args.vla_stride, 400, str(tmp_root), hw, hf, hl, en, nb, sp7, q2l, sm7, ihr, br_r))

    print(f"[COLLECT {args.subtask}] Launching {len(jobs)} rollouts across {args.workers} workers...", flush=True)
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(processes=args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_worker_collect, jobs), 1):
            results.append(res)
            if k % 50 == 0 or k == len(jobs):
                n_ok = sum(1 for r in results if r["success"])
                n_saf = sum(1 for r in results if r["safety"])
                print(f"  [SIM {k}/{len(jobs)}] ok={n_ok}/{k} ({n_ok/k*100:.1f}%) saf={n_saf} elapsed={time.time()-t0:.1f}s", flush=True)

    shutil.rmtree(tmp_root, ignore_errors=True)

    # Keep all successful trajectories + up to 15% failures (for negative Q contrast)
    succ_eps = [r for r in results if r["success"]]
    fail_eps = [r for r in results if not r["success"]][: max(10, len(succ_eps) // 8)]
    kept = succ_eps + fail_eps

    all_vla_frames = []
    ep_offsets = []
    for ep in kept:
        ep_offsets.append(len(all_vla_frames))
        all_vla_frames.extend(ep["vla_frames"])

    print(f"[ENCODE] Kept {len(succ_eps)} succ + {len(fail_eps)} fail eps -> {len(all_vla_frames)} VLA frames to batch-encode...", flush=True)
    device = torch.device("cuda")
    phase_idx = 4 if args.subtask == "p4_insert" else 7
    provider = r2.Stage1RLTFeatureProvider(
        Path(args.stage1_config_dir), args.stage1_config_name, args.stage1_ckpt, device, rlt_adapter_pt=args.rlt_adapter_pt
    )
    z_all, ref_all = batch_encode_vla_frames(provider, all_vla_frames, phase_idx, batch_size=args.batch_size)

    # Pack transitions compactly as contiguous numpy arrays inside a single .pt file!
    trs_flat = []
    for ep, off in zip(kept, ep_offsets):
        for tr in ep["trs"]:
            tr["vla_g"] = off + tr["vla_idx"]
            tr["nvla_g"] = off + tr["nvla_idx"]
            trs_flat.append(tr)

    M = len(trs_flat)
    pack = dict(
        subtask=args.subtask,
        hires_wrist_roi=bool(args.hires_wrist_roi),
        hires_front_roi=bool(args.hires_front_roi),
        hires_left_roi=bool(args.hires_left_roi),
        exact_nullspace=bool(args.exact_nullspace),
        noback_p4=bool(args.noback_p4),
        num_eps=len(kept),
        num_succ_eps=len(succ_eps),
        num_transitions=M,
        vla_stride=args.vla_stride,
        z_table=z_all,           # (V, 2048) fp16
        ref_table=ref_all,       # (V, 32) fp32
        vla_idx=np.array([t["vla_g"] for t in trs_flat], dtype=np.int32),
        nvla_idx=np.array([t["nvla_g"] for t in trs_flat], dtype=np.int32),
        p=np.stack([t["p"] for t in trs_flat]).astype(np.float32),
        w=np.stack([t["w"] for t in trs_flat]).astype(np.uint8),
        f=np.stack([t["f"] for t in trs_flat]).astype(np.uint8),
        g=np.stack([t["g"] for t in trs_flat]).astype(np.float32),
        a=np.stack([t["a"] for t in trs_flat]).astype(np.float32),
        r=np.array([t["r"] for t in trs_flat], dtype=np.float32),
        np_=np.stack([t["np"] for t in trs_flat]).astype(np.float32),
        nw=np.stack([t["nw"] for t in trs_flat]).astype(np.uint8),
        nf=np.stack([t["nf"] for t in trs_flat]).astype(np.uint8),
        ng=np.stack([t["ng"] for t in trs_flat]).astype(np.float32),
        done=np.array([t["done"] for t in trs_flat], dtype=np.float32),
        mc=np.array([t["mc"] for t in trs_flat], dtype=np.float32),
        ep_succ=np.array([t["ep_succ"] for t in trs_flat], dtype=bool),
        is_expert=np.array([t["is_expert"] for t in trs_flat], dtype=bool),
    )
    if args.hires_left_roi:
        pack["l"] = np.stack([t["l"] for t in trs_flat]).astype(np.uint8)
        pack["nl"] = np.stack([t["nl"] for t in trs_flat]).astype(np.uint8)
    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, out_pt, pickle_protocol=4)
    mb = out_pt.stat().st_size / (1024 * 1024)
    print(f"[DONE] Saved {out_pt} ({mb:.1f} MB): {M} transitions from {len(succ_eps)}/{len(kept)} succ eps in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
