# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Unified Stage 1 pi0.5 + Stage 2 Stereo-RLT Persistent Compliant q_tgt Full-Cycle Evaluator.

Strictly zero privileged pose, pure friction MuJoCo 4.0, teacher_planner_used=False, attachment_active=False.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import av
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
from eval_v2_fast import SafePhase12SwitcherV2, select_v2_rows
from p4_microscope_utils import apply_p4_microscope
from evaluate_closed_loop_phase12 import DEFAULT_PHASE_EXEC_STEPS as P12_EXEC_STEPS
from evaluate_closed_loop_phase13 import parse_phase_int_map
from rlt_stage2_residual_ac import (
    CartesianResidualActor,
    Stage1RLTFeatureProvider,
    close_hires_wrist_renderer,
    extract_front_96,
    extract_left_96,
    extract_wrist_128,
    step_persistent_qtgt_dq,
)
import robot_contact_utils as rcu
import usb_rel256_phase12 as usb12
import usb_rel256_phase13 as usb13
from eval_v2_fast import infer_chunk, load_model_fp32

CAMERAS = ("front", "left_wrist", "wrist")


def load_stage2_actor(ckpt_path: str | None, device: torch.device):
    if not ckpt_path:
        return None, None
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    use_left = bool(a.get("hires_left_roi", False))
    act = CartesianResidualActor(use_left_wrist=use_left).to(device).eval()
    act.load_state_dict(ck["actor"], strict=True)
    cfg = dict(
        scale_xyz_m=float(a.get("scale_xyz_mm", 1.8)) * 1e-3,
        scale_rot_rad=float(a.get("scale_rot_rad", 0.022)),
        q_clamp_rad=float(a.get("q_clamp_rad", 0.028)),
        vla_stride=int(a.get("vla_stride", 6)),
        flow_m=int(a.get("flow_m", 1)),
        hires_wrist_roi=bool(a.get("hires_wrist_roi", False)),
        hires_front_roi=bool(a.get("hires_front_roi", False)),
        hires_left_roi=use_left,
        exact_nullspace=bool(a.get("exact_nullspace", False)),
    )
    return act, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--stage1-config-dir", default=str(EXP12 / "configs"))
    ap.add_argument("--stage1-config-name", default="phase12_sft8")
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--rlt-adapter-pt", default="")
    ap.add_argument("--p4-vla-ckpt", default="")
    ap.add_argument("--p4-vla-config-dir", default="")
    ap.add_argument("--p4-vla-config-name", default="pi05_p4_recovery_v2")
    ap.add_argument("--p7-vla-ckpt", default="")
    ap.add_argument("--p7-vla-config-dir", default="")
    ap.add_argument("--p7-vla-config-name", default="p8_vision_sft8")
    ap.add_argument("--tail-p13-ckpt", default="")
    ap.add_argument("--tail-p13-config-dir", default=str(EXP13 / "configs"))
    ap.add_argument("--tail-p13-config-name", default="phase13_sft8")
    ap.add_argument("--p4-rlt-ckpt", default="")
    ap.add_argument("--p56-rlt-ckpt", default="")
    ap.add_argument("--p5-only", action="store_true")
    ap.add_argument("--p7-rlt-ckpt", default="")
    ap.add_argument("--scenes-root", default=str(H2 / "scenes_v2_keepvis"))
    ap.add_argument("--scenes-json", default="")
    ap.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--limit", type=int, default=34)
    ap.add_argument("--start-snapshot", default="cycle_phase_raise_to_pregrasp.npz")
    ap.add_argument("--start-phase", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--phase-flow-m", default="4:1")
    ap.add_argument("--phase-exec-steps", default="")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--microscope-p4", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    provider = Stage1RLTFeatureProvider(Path(args.stage1_config_dir), args.stage1_config_name, args.stage1_ckpt, device, rlt_adapter_pt=args.rlt_adapter_pt)
    p4_provider = Stage1RLTFeatureProvider(Path(args.p4_vla_config_dir or args.stage1_config_dir), args.p4_vla_config_name, args.p4_vla_ckpt, device, rlt_adapter_pt=args.rlt_adapter_pt) if args.p4_vla_ckpt else provider
    p7_provider = Stage1RLTFeatureProvider(Path(args.p7_vla_config_dir or args.stage1_config_dir), args.p7_vla_config_name, args.p7_vla_ckpt, device, rlt_adapter_pt=args.rlt_adapter_pt) if args.p7_vla_ckpt else provider
    tail_p13 = None
    p13_stream = torch.cuda.Stream(priority=0)
    if args.tail_p13_ckpt:
        usb13.register()
        tail_p13 = load_model_fp32(Path(args.tail_p13_config_dir), args.tail_p13_config_name, args.tail_p13_ckpt)
    p4_actor, p4_cfg = load_stage2_actor(args.p4_rlt_ckpt, device)
    p7_actor, p7_cfg = load_stage2_actor(args.p7_rlt_ckpt, device)
    p56_actor = p56_cfg = None
    if args.p56_rlt_ckpt:
        from train_stage2_p56_rlt import Stage2P56ResidualActor, make_p56_feat
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
    flow_m_map = parse_phase_int_map(args.phase_flow_m)
    exec_steps_map = parse_phase_int_map(args.phase_exec_steps)

    dest = H2 / "eval" / args.label
    dest.mkdir(parents=True, exist_ok=True)
    if args.scenes_json:
        raw_sc = json.loads(Path(args.scenes_json).read_text())
        if args.seeds:
            ord_m = {s: i for i, s in enumerate(args.seeds)}
            raw_sc = [s for s in raw_sc if int(s["seed"]) in ord_m]
            raw_sc.sort(key=lambda s: ord_m[int(s["seed"])])
        elif args.limit < len(raw_sc):
            idxs = np.linspace(0, len(raw_sc) - 1, min(args.limit, len(raw_sc)), dtype=int)
            raw_sc = [raw_sc[i] for i in idxs]
        rows = [dict(scene=s, record=None) for s in raw_sc]
    else:
        rows = select_v2_rows(args.split, args.seeds, args.limit, Path(args.scenes_root))
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_idx]

    reports = []
    t_all = time.time()
    for row in rows:
        t0 = time.time()
        sc = row["scene"]
        seed = int(sc["seed"])
        rec_path = Path(row["record"]["path"]) if row.get("record") else None
        run_dir = dest / str(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        env = PolicyExecutor(Path(sc["scene"]), run_dir, seed, brightness=float(sc["brightness"]))
        video = None
        stream = None
        try:
            if rec_path is not None and (rec_path / "demonstration_full.npz").exists():
                with np.load(rec_path / "demonstration_full.npz", allow_pickle=True) as d:
                    q_home0_demo = d["observation.state"][0, 6:13].astype(np.float32)
                env.reset(snapshot=rec_path / args.start_snapshot)
            else:
                q_home0_demo = None
                env.reset()
            task = env._task
            obs0 = env.observe()
            q_home0 = obs0["observation.state"][6:13].astype(np.float32).copy() if args.start_phase == 0 else q_home0_demo

            if args.save_video:
                video = av.open(str(run_dir / "rollout.mp4"), mode="w")
                stream = video.add_stream("h264", rate=25)
                stream.width = 224 * 3
                stream.height = 224
                stream.pix_fmt = "yuv420p"

            sw = SafePhase12SwitcherV2(start_phase=args.start_phase, max_grasp_retries=2)
            cur_phase = args.start_phase
            pending: list[np.ndarray] = []
            cached_z = cached_ref_rel32 = None
            steps_since_vla = 999
            phase_step = 0
            p4_unjam_retries = 0
            p4_unjam_lift = 0
            q_tgt: np.ndarray | None = None
            pending_dq: list[np.ndarray] = []

            raw0 = obs0["observation.state"][:14].astype(np.float32)
            tcp_entry = raw0[:3].copy()
            tcp_grasp_mem = raw0[:3].copy()
            q_p2_lift_mem = raw0[6:13].copy()
            p0_q_trace: list[np.ndarray] = [raw0[6:13].copy()]
            p10_waypoints: list[np.ndarray] = [q_home0.copy()]
            p10_wp_idx = 0
            last_cmd_q7 = raw0[6:13].copy()
            q_hist = [raw0[6:13].copy()] * 5
            tcp_z_hist = [float(raw0[2])] * 6
            tau_p4_free = np.asarray(task.d.actuator_force[rcu.ARM_ACT_SLICE], dtype=np.float64).copy()
            prev_act6 = np.zeros(6, dtype=np.float32)
            stall_steps = 0
            prev_fq_min = 0.050

            min_lat_p4 = float("inf")
            max_dep_p4 = float("-inf")
            min_holder_dz = float("inf")
            min_home_err = float("inf")
            step_done = 0

            for i in range(args.max_steps):
                obs = env.observe()
                raw_st = obs["observation.state"]
                if cur_phase == 0:
                    p0_q_trace.append(raw_st[6:13].astype(np.float32).copy())
                if i > 0:
                    q_cur_f = raw_st[6:13].astype(np.float32)
                    q_hist.append(q_cur_f.copy())
                    if len(q_hist) > 5:
                        q_hist.pop(0)
                    tcp_z_hist.append(float(raw_st[2]))
                    if len(tcp_z_hist) > 6:
                        tcp_z_hist.pop(0)
                    stall_steps = (stall_steps + 1) if float(np.linalg.norm(q_cur_f - q_hist[0])) < 0.0020 else 0
                fq_margin = np.asarray(task.d.qpos[task.fq] - task.finger_lower, dtype=np.float32)
                fq_min_now = float(np.min(fq_margin))
                fq_min_vel = fq_min_now - prev_fq_min
                fq_slam_guard = (fq_min_vel < -0.0005) and ((fq_min_now + fq_min_vel) < 0.00010)
                prev_fq_min = fq_min_now
                st21 = usb12.make_state21(raw_st, q_home0)
                new_phase, cmd_ov = sw.step_update(i, st21, fq_margin, q_home0)
                if new_phase != cur_phase:
                    if new_phase == 2 or (cur_phase == 7 and new_phase == 8):
                        tcp_grasp_mem = raw_st[:3].astype(np.float32).copy()
                    if cur_phase == 2 and new_phase == 3:
                        q_p2_lift_mem = raw_st[6:13].astype(np.float32).copy()
                    if new_phase == 10:
                        wps_rev: list[np.ndarray] = []
                        last_wp: np.ndarray | None = None
                        for q_pt in reversed(p0_q_trace):
                            qf = np.asarray(q_pt, dtype=np.float32)
                            if last_wp is None or float(np.max(np.abs(qf - last_wp))) >= 0.05:
                                wps_rev.append(qf.copy())
                                last_wp = qf
                        wps_rev.append(q_home0.copy())
                        p10_waypoints = wps_rev
                        p10_wp_idx = 0
                    cur_phase = new_phase
                    pending.clear()
                    pending_dq.clear()
                    cached_z = cached_ref_rel32 = None
                    steps_since_vla = 999
                    phase_step = 0
                    p4_unjam_retries = 0
                    p4_unjam_lift = 0
                    q_tgt = raw_st[6:13].astype(np.float64).copy()
                    tcp_entry = raw_st[:3].astype(np.float32).copy()
                    last_cmd_q7 = raw_st[6:13].astype(np.float32).copy()
                    q_hist = [raw_st[6:13].astype(np.float32).copy()] * 5
                    tcp_z_hist = [float(raw_st[2])] * 6
                    tau_p4_free = np.asarray(task.d.actuator_force[rcu.ARM_ACT_SLICE], dtype=np.float64).copy()
                    prev_act6 = np.zeros(6, dtype=np.float32)
                    stall_steps = 0

                if cur_phase == 4:
                    met = task.insertion_metrics()
                    min_lat_p4 = min(min_lat_p4, float(met["lateral_error_m"] * 1000.0))
                    max_dep_p4 = max(max_dep_p4, float(met["depth_m"] * 1000.0))
                elif cur_phase == 7:
                    tip, _ = task.object_pose()
                    min_holder_dz = min(min_holder_dz, float(abs(tip[2] - task.initial_tip[2]) * 1000.0))
                elif cur_phase == 11:
                    min_home_err = min(min_home_err, float(np.linalg.norm(raw_st[6:13] - q_home0)))

                use_actor = None
                use_cfg = None
                subtask_name = ""
                if cur_phase == 4 and p4_actor is not None:
                    use_actor, use_cfg, subtask_name = p4_actor, p4_cfg, "p4_insert"
                elif cur_phase == 7 and p7_actor is not None:
                    use_actor, use_cfg, subtask_name = p7_actor, p7_cfg, "p7_holder"

                if cmd_ov is not None:
                    cmd = cmd_ov
                    pending.clear()
                    pending_dq.clear()
                elif p56_actor is not None and (
                    cur_phase == 5
                    or ((not args.p5_only) and (cur_phase == 6 or (cur_phase in (10, 11) and p56_cfg["arch_version"] == "p56_rlt_v2")))
                ):
                    if q_tgt is None:
                        q_tgt = raw_st[6:13].astype(np.float64).copy()
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
                            dq = step_persistent_qtgt_dq(
                                task, a7[:6],
                                scale_xyz_m=p56_cfg["p5_scale_xyz_m"],
                                scale_rot_rad=p56_cfg["p5_scale_rot_rad"],
                                max_dq_rad=0.025,
                                exact_nullspace=True,
                            )
                            pending_dq = [0.5 * dq, 0.5 * dq]
                        q_cur = raw_st[6:13].astype(np.float64)
                        q_tgt = q_cur + np.clip((q_tgt + pending_dq.pop(0)) - q_cur, -p56_cfg["p5_q_clamp_rad"], p56_cfg["p5_q_clamp_rad"])
                        cmd = np.zeros(8, dtype=np.float32)
                        cmd[:7] = q_tgt.astype(np.float32)
                        cmd[7] = 0.98
                        if (fq_min_now < 0.00018 and float(np.sum(fq_margin)) < 0.00085) or fq_min_now < 0.00010 or fq_slam_guard:
                            cmd[7] = 0.0
                        steps_since_vla += 1
                    else:
                        if cached_z is None or steps_since_vla >= 12:
                            cached_z, _, _ = provider.infer(obs, q_home0, cur_phase, step_seed=7317 + i, flow_m=1)
                            steps_since_vla = 0
                        q_cur = raw_st[6:13].astype(np.float64)
                        if cur_phase == 6:
                            q_goal_use = q_p2_lift_mem
                        elif cur_phase == 11:
                            q_goal_use = q_home0
                        else:
                            if p10_wp_idx < len(p10_waypoints) - 1 and float(np.max(np.abs(q_cur - p10_waypoints[p10_wp_idx]))) < 0.04:
                                p10_wp_idx += 1
                            q_goal_use = p10_waypoints[p10_wp_idx]
                        max_dq_use = p56_cfg["max_dq_p6"]
                        f56 = make_p56_feat(6, raw_st[6:13], q_goal_use, raw_st[:3], tcp_entry, max_dq_joint=max_dq_use)
                        with torch.no_grad():
                            a7 = p56_actor(
                                torch.from_numpy(cached_z[None]).to(device),
                                torch.from_numpy(f56[None]).to(device),
                            )[0].cpu().numpy().astype(np.float32)
                        if cur_phase == 10 and (p10_wp_idx < len(p10_waypoints) - 1 or float(np.max(np.abs(q_cur - q_home0))) > 0.16):
                            dq_step = np.clip((a7.astype(np.float64) / 0.96) * 0.030, -0.030, 0.030)
                            q_tgt = q_cur + np.clip((q_tgt + dq_step) - q_cur, -0.045, 0.045)
                            q_cmd = q_tgt
                        else:
                            dq_step = np.clip((a7.astype(np.float64) / 0.96) * max_dq_use, -max_dq_use, max_dq_use)
                            q_cmd = q_cur + dq_step
                            q_tgt = q_cmd.copy()
                        cmd = np.zeros(8, dtype=np.float32)
                        cmd[:7] = q_cmd.astype(np.float32)
                        cmd[7] = 0.98 if cur_phase == 6 else 0.0
                        if cmd[7] >= 0.85 and ((fq_min_now < 0.00018 and float(np.sum(fq_margin)) < 0.00085) or fq_min_now < 0.00010 or fq_slam_guard):
                            cmd[7] = 0.0
                        steps_since_vla += 1
                elif use_actor is not None:
                    if q_tgt is None:
                        q_tgt = raw_st[6:13].astype(np.float64).copy()
                    # Non-privileged P4 rim-unjam retry: if stuck ABOVE the socket rim
                    # (never after descending >= 15mm into the socket!), lift straight up ~12mm via
                    # robot URDF Jacobian and reset phase_step for a clean visual re-alignment.
                    p4_above_rim = sw.p4_entry_z is None or (sw.p4_entry_z - float(raw_st[2])) < 0.015
                    if cur_phase == 4 and p4_unjam_lift == 0 and p4_unjam_retries < 2 and phase_step >= 240 and sw.settle_steps < 5 and p4_above_rim:
                        p4_unjam_retries += 1
                        p4_unjam_lift = 12
                        pending_dq.clear()
                        cached_z = cached_ref_rel32 = None
                        steps_since_vla = 999
                        phase_step = 0
                        stall_steps = 0
                        prev_act6 = np.zeros(6, dtype=np.float32)
                    if cur_phase == 4 and p4_unjam_lift > 0:
                        p4_unjam_lift -= 1
                        lift_act = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        dq_up = step_persistent_qtgt_dq(
                            task, lift_act, scale_xyz_m=0.0012, scale_rot_rad=0.0, max_dq_rad=0.020, exact_nullspace=True
                        )
                        q_cur = raw_st[6:13].astype(np.float64)
                        q_tgt = q_cur + np.clip((q_tgt + dq_up) - q_cur, -0.020, 0.020)
                        cmd = np.zeros(8, dtype=np.float32)
                        cmd[:7] = q_tgt.astype(np.float32)
                        cmd[7] = 0.98
                        if (fq_min_now < 0.00018 and float(np.sum(fq_margin)) < 0.00085) or fq_min_now < 0.00010 or fq_slam_guard:
                            cmd[7] = 0.0
                        if p4_unjam_lift == 0:
                            q_tgt = raw_st[6:13].astype(np.float64).copy()
                            tcp_entry = raw_st[:3].astype(np.float32).copy()
                            sw.p4_entry_z = float(raw_st[2])
                    else:
                        if not pending_dq:
                            cur_prov = p4_provider if cur_phase == 4 else (p7_provider if cur_phase == 7 else provider)
                            if cached_z is None or steps_since_vla >= use_cfg["vla_stride"]:
                                cached_z, _, cached_ref_rel32 = cur_prov.infer(
                                    obs, q_home0, cur_phase, step_seed=7317 + phase_step, flow_m=use_cfg["flow_m"]
                                )
                                steps_since_vla = 0
                            w_crop = extract_wrist_128(task, obs, hires_wrist=use_cfg["hires_wrist_roi"])
                            f_crop = extract_front_96(task, obs, subtask_name, hires_front=use_cfg["hires_front_roi"])
                            l_crop = extract_left_96(task, obs, subtask_name, hires_left=use_cfg["hires_left_roi"])
                            p_feat = rcu.extract_contact_proprio(
                                task, raw_st, last_cmd_q7, q_hist[0], tcp_entry, stall_steps, prev_act6
                            )
                            with torch.no_grad():
                                l_t = torch.from_numpy(l_crop[None]).to(device) if l_crop is not None else None
                                act6 = use_actor(
                                    torch.from_numpy(cached_z[None]).to(device),
                                    torch.from_numpy(p_feat[None]).to(device),
                                    torch.from_numpy(cached_ref_rel32[None]).to(device),
                                    torch.from_numpy(w_crop[None]).to(device),
                                    torch.from_numpy(f_crop[None]).to(device),
                                    l_t,
                                )[0].cpu().numpy().astype(np.float32)
                            dq = step_persistent_qtgt_dq(
                                task, act6,
                                scale_xyz_m=use_cfg["scale_xyz_m"],
                                scale_rot_rad=use_cfg["scale_rot_rad"],
                                max_dq_rad=0.025,
                                exact_nullspace=use_cfg["exact_nullspace"],
                            )
                            prev_act6 = act6.copy()
                            pending_dq = [0.5 * dq, 0.5 * dq]
                        q_cur = raw_st[6:13].astype(np.float64)
                        q_cl = float(use_cfg["q_clamp_rad"])
                        dq_step_p = pending_dq.pop(0)
                        if cur_phase == 4:
                            # Real-robot motor-torque Jacobian admittance compliance (100% robot-only SDK2 torques + URDF J):
                            # normal descent & leaf-spring seating stay <= 3.5 N (full authority); smooth bleed above 5.5 N
                            # prevents rigid bottom-stop spikes from exceeding the 25 N guard without starving spring entry.
                            tau_cur = np.asarray(task.d.actuator_force[rcu.ARM_ACT_SLICE], dtype=np.float64)
                            J_rob = rcu.compute_robot_tcp_jacobian(task)
                            f_est3 = np.linalg.lstsq(J_rob.T, -(tau_cur - tau_p4_free), rcond=1e-2)[0][:3]
                            f_est_n = float(np.linalg.norm(f_est3))
                            if f_est_n > 5.5:
                                q_tgt = q_cur + 0.80 * (q_tgt - q_cur)
                                dq_step_p = np.zeros_like(dq_step_p)
                                q_cl = min(q_cl, 0.0090)
                        q_tgt = q_cur + np.clip((q_tgt + dq_step_p) - q_cur, -q_cl, q_cl)
                        cmd = np.zeros(8, dtype=np.float32)
                        cmd[:7] = q_tgt.astype(np.float32)
                        cmd[7] = 0.98
                        fq_min_thr = 0.00014 if cur_phase == 7 else 0.00010
                        if (fq_min_now < 0.00018 and float(np.sum(fq_margin)) < 0.00085) or fq_min_now < fq_min_thr or fq_slam_guard:
                            cmd[7] = 0.0
                        steps_since_vla += 1
                        phase_step += 1
                else:
                    if not pending:
                        if tail_p13 is not None and cur_phase >= 8:
                            m13, tx13, un13 = tail_p13
                            raw13 = dict(
                                state=usb13.make_state24(raw_st, q_home0, tcp_grasp_mem),
                                image=obs["observation.images.front"],
                                left_wrist_image=obs["observation.images.left_wrist"],
                                wrist_image=obs["observation.images.wrist"],
                                prompt=usb13.PHASE_PROMPTS[cur_phase + 1],
                            )
                            ref_abs = infer_chunk(m13, tx13, un13, raw13, i, flow_m=int(flow_m_map.get(cur_phase, 1)), stream=p13_stream)
                            k_exec = min(int(exec_steps_map.get(cur_phase, P12_EXEC_STEPS[cur_phase])), len(ref_abs))
                            pending = list(ref_abs[:k_exec])
                        else:
                            cur_prov = p4_provider if cur_phase == 4 else (p7_provider if cur_phase == 7 else provider)
                            obs_in = obs
                            if cur_phase == 4 and args.microscope_p4:
                                im_f, im_lw, im_w = apply_p4_microscope(obs["observation.images.front"], obs["observation.images.wrist"])
                                obs_in = dict(obs)
                                obs_in["observation.images.front"] = im_f
                                obs_in["observation.images.left_wrist"] = im_lw
                                obs_in["observation.images.wrist"] = im_w
                            raw_in = dict(
                                state=st21,
                                image=obs_in["observation.images.front"],
                                left_wrist_image=obs_in["observation.images.left_wrist"],
                                wrist_image=obs_in["observation.images.wrist"],
                                prompt=usb12.PHASE_PROMPTS[cur_phase],
                            )
                            ref_abs = infer_chunk(
                                cur_prov.vla,
                                cur_prov.tx,
                                cur_prov.un,
                                raw_in,
                                i,
                                flow_m=int(flow_m_map.get(cur_phase, 1)),
                                stream=p13_stream,
                            )
                            default_k = min(4, P12_EXEC_STEPS[cur_phase]) if cur_phase <= 6 else P12_EXEC_STEPS[cur_phase]
                            k_exec = min(int(exec_steps_map.get(cur_phase, default_k)), len(ref_abs))
                            pending = list(ref_abs[:k_exec])
                    cmd = np.asarray(pending.pop(0), dtype=np.float32).copy()
                    if cmd[7] >= 0.85 and ((float(np.min(fq_margin)) < 0.00018 and float(np.sum(fq_margin)) < 0.00085) or float(np.min(fq_margin)) < 0.00010):
                        cmd[7] = 0.0

                res = env.step(cmd)
                step_done += 1
                last_cmd_q7 = cmd[:7].copy()

                if video is not None:
                    panel = np.concatenate([obs["observation.images." + c] for c in CAMERAS], axis=1)
                    frame = av.VideoFrame.from_ndarray(panel, format="rgb24")
                    for pkt in stream.encode(frame):
                        video.mux(pkt)
                if res["terminated"] or res["truncated"]:
                    break

            if video is not None:
                for pkt in stream.encode():
                    video.mux(pkt)
                video.close()
                video = None

            out = env.outcomes()
            phys = out["physical"]
            rep = dict(
                seed=seed, split=args.split, steps=step_done, elapsed_s=round(time.time() - t0, 1),
                success=bool(phys.get("success", False)),
                events=phys.get("event_times", {}),
                min_lat_p4_mm=round(min_lat_p4, 3) if np.isfinite(min_lat_p4) else None,
                max_dep_p4_mm=round(max_dep_p4, 3) if np.isfinite(max_dep_p4) else None,
                min_holder_dz_mm=round(min_holder_dz, 3) if np.isfinite(min_holder_dz) else None,
                min_home_err_rad=round(min_home_err, 5) if np.isfinite(min_home_err) else None,
                final_phase=int(cur_phase),
                grasp_retries=sw.grasp_retries,
                empty_grasp_events=sw.empty_grasp_events,
                invalid_reason=phys.get("invalid_reason"),
                reason=out.get("reason"),
                teacher_planner_used=bool(out.get("teacher_planner_used", False)),
                attachment_active=bool(out.get("attachment_active", False)),
                phase_transitions=sw.transitions,
            )
            (run_dir / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
            print(json.dumps({k: v for k, v in rep.items() if k != "phase_transitions"}), flush=True)
            reports.append(rep)
        finally:
            if video is not None:
                try:
                    video.close()
                except Exception:
                    pass
            close_hires_wrist_renderer(env._task)
            env.close()

    ev_names = ("lift", "insert", "withdraw", "return", "release", "home")
    counts = {e: sum(1 for r in reports if e in r["events"]) for e in ev_names}
    summary = dict(
        completed=True, label=args.label, split=args.split, total=len(reports),
        full_success=sum(1 for r in reports if r["success"]),
        safety_stops=sum(1 for r in reports if r["invalid_reason"] == "safety_stop"),
        event_counts=counts, elapsed_total_s=round(time.time() - t_all, 1), reports=reports,
    )
    fname = "summary.json" if args.num_shards == 1 else f"summary_shard{args.shard_idx}.json"
    (dest / fname).write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY %s (%s): full=%d/%d safety=%d events=%s time=%.1fs" % (
        args.label, args.split, summary["full_success"], summary["total"], summary["safety_stops"], counts, summary["elapsed_total_s"]
    ), flush=True)


if __name__ == "__main__":
    main()
