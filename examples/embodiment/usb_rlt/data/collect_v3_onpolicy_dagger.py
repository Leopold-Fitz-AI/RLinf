# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1B On-Policy DAgger Collector & Dataset Merger (strictly train-only, zero val/test overlap).

Subcommands:
  1) rollout: Sharded GPU rollout of Stage 1A policy from Home on N train scenes, saving real
              on-policy MuJoCo state snapshots at P1/P4/P7 entry & stall points.
  2) recover: Parallel CPU/EGL smooth CleanSmoothTeacher recovery (speedup=1.5, weld=False, dynamic
              in-hand recalibration) from all harvested on-policy snapshots.
  3) merge:   Merge train_v3_fresh + verified on-policy DAgger episodes (2x weighted) into
              /mlplatform/jy/v3_fresh_nvme/train_v3_fresh_plus_dagger (symlinking existing NVMe videos).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import shutil
import sys
import time
from pathlib import Path

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

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.common.datasets.utils import create_empty_dataset_info

from collect_and_build_v3_fresh_nvme import (
    CAMERAS,
    MOTION_SPEEDUP,
    _close_renderers,
    _recalib,
    _save_surgical_episode,
    make_teacher,
    numeric_stats,
)
import usb_rel256_phase12 as usb12

NVME_ROOT = Path("/mlplatform/jy/v3_fresh_nvme")
SCENES_MANIFEST = NVME_ROOT / "train_scenes_v3_1200.json"
RAW_FULL = NVME_ROOT / "raw_full_1200"
DAGGER_ROOT = NVME_ROOT / "dagger_s1b"
BASE_DS = NVME_ROOT / "train_v3_fresh_bal"
MERGED_DS = NVME_ROOT / "train_v3_fresh_plus_dagger"


def _forbidden_seeds() -> set[int]:
    forb: set[int] = set()
    for p in (
        H2 / "scenes_v3_real/stage2_validation_scenes_v3.json",
        H2 / "scenes_v3_real/stage2_test_scenes_v3.json",
    ):
        if p.exists():
            forb |= {int(s["seed"]) for s in json.loads(p.read_text())}
    for mf in (EXP13 / "validation_manifest.json", EXP13 / "test_manifest.json"):
        if mf.exists():
            forb |= {int(r["parent_scene_seed"]) for r in json.loads(mf.read_text())}
    return forb


def cmd_rollout(args: argparse.Namespace) -> None:
    import torch
    from eval_v2_fast import SafePhase12SwitcherV2
    from rlt_stage2_residual_ac import Stage1RLTFeatureProvider
    from policy_executor import PolicyExecutor

    forbidden = _forbidden_seeds()
    train_scenes = json.loads(SCENES_MANIFEST.read_text())
    # Keep only scenes with valid Pass 1 full_cycle snapshots so we can restore Home identically
    eligible = []
    for sc in train_scenes:
        sd = int(sc["seed"])
        assert sd not in forbidden, f"LEAKAGE seed={sd}"
        ep_dir = RAW_FULL / str(sd)
        if (ep_dir / "v3f_done.json").exists() and (ep_dir / "cycle_phase_raise_to_pregrasp.npz").exists():
            eligible.append((sc, ep_dir))
    step = max(1, len(eligible) // max(1, args.limit))
    selected = eligible[::step][: args.limit]
    my_rows = [r for i, r in enumerate(selected) if i % args.num_shards == args.shard_idx]

    snap_dir = DAGGER_ROOT / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    prov = Stage1RLTFeatureProvider(Path(args.config_dir), args.config_name, args.checkpoint, device)

    harvested: list[dict] = []
    t0 = time.time()
    for idx_i, (sc, rec_dir) in enumerate(my_rows, 1):
        seed = int(sc["seed"])
        run_dir = Path("/dev/shm/jy_h2_tmp") / f"dag_roll_{os.getpid()}_{seed}"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        env = PolicyExecutor(Path(sc["scene"]), run_dir, seed, brightness=float(sc["brightness"]))
        try:
            env.reset(snapshot=rec_dir / "cycle_phase_raise_to_pregrasp.npz")
            task = env._task
            obs = env.observe()
            q_home0 = obs["observation.state"][6:13].astype(np.float32).copy()
            sw = SafePhase12SwitcherV2(start_phase=0, max_grasp_retries=2)
            pending: list[np.ndarray] = []
            prev_phase = -1
            saved_flags = {
                "p1_retry": False,
                "p1_stall": False,
                "p3_stall": False,
                "p4_entry": False,
                "p4_stall": False,
                "p7_entry": False,
                "p7_stall": False,
            }
            p1_steps = 0
            p3_steps = 0
            p4_steps = 0
            p7_steps = 0

            def save_snap(tag: str, subtask: str, step_i: int, extra: dict) -> None:
                snap_name = f"s{seed}_{tag}"
                prev_out = task.out
                task.out = snap_dir
                try:
                    task.snapshot(snap_name)
                finally:
                    task.out = prev_out
                fn = snap_dir / f"{snap_name}.npz"
                assert fn.exists(), f"Missing saved snapshot {fn}"
                harvested.append(dict(
                    job_id=f"dag_{subtask}_{seed}_{tag}",
                    subtask=subtask,
                    tag=tag,
                    seed=seed,
                    scene=sc,
                    rec_path=str(rec_dir),
                    snapshot=str(fn),
                    q_home0=q_home0.tolist(),
                    step=step_i,
                    **extra,
                ))

            for step_i in range(args.max_steps):
                obs = env.observe()
                raw_st = obs["observation.state"].astype(np.float32)
                st21 = usb12.make_state21(raw_st, q_home0)
                fq_margin = task.d.qpos[task.fq] - task.finger_lower
                cur_phase, cmd_ov = sw.step_update(step_i, st21, fq_margin, q_home0)
                if cur_phase != prev_phase:
                    pending = []
                    prev_phase = cur_phase
                    if cur_phase == 1 and sw.grasp_retries > 0 and not saved_flags["p1_retry"]:
                        save_snap("p1_retry", "p1", step_i, dict(retries=int(sw.grasp_retries)))
                        saved_flags["p1_retry"] = True
                    elif cur_phase == 4 and not saved_flags["p4_entry"]:
                        m = task.insertion_metrics()
                        if m["lateral_error_m"] >= 0.0004:
                            save_snap("p4_entry", "p4", step_i, dict(lat_mm=round(m["lateral_error_m"] * 1e3, 2), dep_mm=round(m["depth_m"] * 1e3, 2)))
                            saved_flags["p4_entry"] = True
                    elif cur_phase == 7 and not saved_flags["p7_entry"]:
                        save_snap("p7_entry", "p7", step_i, {})
                        saved_flags["p7_entry"] = True

                if cur_phase == 1:
                    p1_steps += 1
                    if p1_steps == 420 and not saved_flags["p1_stall"]:
                        save_snap("p1_stall", "p1", step_i, dict(p1_steps=p1_steps))
                        saved_flags["p1_stall"] = True
                    if p1_steps >= 520:
                        break
                elif cur_phase == 3:
                    p3_steps += 1
                    if p3_steps == 220 and not saved_flags["p3_stall"]:
                        m = task.insertion_metrics()
                        save_snap("p3_stall", "p3", step_i, dict(lat_mm=round(m["lateral_error_m"] * 1e3, 2), dep_mm=round(m["depth_m"] * 1e3, 2)))
                        saved_flags["p3_stall"] = True
                    if p3_steps >= 260:
                        break
                elif cur_phase == 4:
                    p4_steps += 1
                    if p4_steps == 120 and not saved_flags["p4_stall"] and "insert" not in task.tracker.event_times:
                        m = task.insertion_metrics()
                        save_snap("p4_stall", "p4", step_i, dict(lat_mm=round(m["lateral_error_m"] * 1e3, 2), dep_mm=round(m["depth_m"] * 1e3, 2)))
                        saved_flags["p4_stall"] = True
                    if p4_steps >= 360 and "insert" not in task.tracker.event_times:
                        break
                elif cur_phase == 7:
                    p7_steps += 1
                    if p7_steps == 120 and not saved_flags["p7_stall"] and "return" not in task.tracker.event_times:
                        save_snap("p7_stall", "p7", step_i, {})
                        saved_flags["p7_stall"] = True
                    if p7_steps >= 240 and "return" not in task.tracker.event_times:
                        break
                elif cur_phase >= 8:
                    break

                if cmd_ov is not None:
                    cmd = cmd_ov
                else:
                    if not pending:
                        _, ref_abs4, _ = prov.infer(obs, q_home0, cur_phase, step_seed=9101 + step_i, flow_m=1)
                        pending = list(ref_abs4[:4])
                    cmd = np.asarray(pending.pop(0), dtype=np.float32).copy()
                    if cmd[7] >= 0.85 and float(np.min(fq_margin)) < 0.00018 and float(np.sum(fq_margin)) < 0.00085:
                        cmd[7] = 0.0

                res = env.step(cmd)
                if res["terminated"] or res["truncated"]:
                    break

            out = env.outcomes()["physical"]
            ev = sorted((out.get("event_times") or {}).keys())
            if idx_i % 10 == 0 or idx_i == len(my_rows):
                print(
                    f"[DAGGER ROLL s{args.shard_idx} {idx_i}/{len(my_rows)}] seed={seed} ph={prev_phase} "
                    f"ev={ev} snaps={len(harvested)} elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )
        finally:
            env.close()
            shutil.rmtree(run_dir, ignore_errors=True)

    out_json = DAGGER_ROOT / f"harvested_shard{args.shard_idx}.json"
    out_json.write_text(json.dumps(harvested, indent=2) + "\n")
    print(f"[DAGGER ROLL DONE s{args.shard_idx}] saved {len(harvested)} snapshots -> {out_json}", flush=True)


def _recover_one(args: tuple[int, dict, list[int]]) -> dict:
    w_idx, item, egl_gpu_ids = args
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_gpu_ids[w_idx % max(1, len(egl_gpu_ids))])
    job_id = item["job_id"]
    subtask = item["subtask"]
    seed = int(item["seed"])
    sc = item["scene"]
    snap = Path(item["snapshot"])
    ep_dir = DAGGER_ROOT / "episodes" / job_id

    if (ep_dir / "meta.json").exists() and (ep_dir / "demonstration_full.npz").exists():
        m = json.loads((ep_dir / "meta.json").read_text())
        if m.get("verified"):
            return dict(job_id=job_id, subtask=subtask, seed=seed, ok=True, cached=True, frames=int(m["frames"]))

    scratch = Path("/dev/shm/jy_h2_tmp") / f"dag_rec_{os.getpid()}_{job_id}"
    t = None
    try:
        t = make_teacher(
            Path(sc["scene"]), scratch, seed, float(sc["brightness"]),
            capture=False, speedup=MOTION_SPEEDUP, smooth=True,
        )
        t.restore(snap)
        t.d.eq_active[t.weld] = False

        if subtask == "p1":
            # Recover exact visual approach & descent from real post-retry above-holder state
            rec_dir = Path(item["rec_path"])
            t_ref = make_teacher(
                Path(sc["scene"]), scratch / "ref", seed, float(sc["brightness"]),
                capture=False, speedup=MOTION_SPEEDUP, smooth=False,
            )
            try:
                t_ref.restore(rec_dir / "cycle_phase_approach_holder.npz")
                snap_p, snap_r = t_ref.pose()
            finally:
                _close_renderers(t_ref)
            p_goal = snap_p - np.array([0.0, 0.0, 0.030])
            above = p_goal + np.array([0.0, 0.0, 0.028])
            t.stage = "approach_holder"
            t.capture = True
            t.full_samples = []
            t.move(above, snap_r, 1.4, settle=0.05)
            t.move(p_goal, snap_r, 2.2, settle=0.15)
            end_p, _ = t.pose()
            err_mm = float(np.linalg.norm(end_p - p_goal) * 1000.0)
            if err_mm > 1.5 or len(t.full_samples) < 25:
                return dict(job_id=job_id, subtask=subtask, seed=seed, ok=False, err=f"p1_err_{err_mm:.2f}")
            for row in t.full_samples:
                row["phase"] = "approach_holder"
            fam = "dagger_p1"
        elif subtask == "p3":
            _recalib(t)
            fq_sum = float(np.sum(t.d.qpos[t.fq] - t.finger_lower))
            if fq_sum < 0.00105:
                return dict(job_id=job_id, subtask=subtask, seed=seed, ok=False, err="not_grasped")
            t.capture = True
            t.full_samples = []
            t.stage = "descend_to_port"
            tip_c, objr_c = t.object_pose()
            # Ensure safe transit clearance above table/dock before aligning to port mouth
            if tip_c[2] < 0.818:
                t.object_move(np.array([tip_c[0], tip_c[1], 0.824]), objr_c, 0.8)
                _recalib(t)
            t.object_move(t.port - t.S[:, 2] * 0.004, t.S, 2.0)
            _recalib(t)
            m1 = t.insertion_metrics()
            if m1["lateral_error_m"] > 0.0025 or len(t.full_samples) < 25:
                return dict(job_id=job_id, subtask=subtask, seed=seed, ok=False, err=f"p3_lat_{m1['lateral_error_m']*1e3:.2f}")
            for row in t.full_samples:
                row["phase"] = "descend_to_port"
            fam = "dagger_p3"
        elif subtask == "p4":
            _recalib(t)
            fq_sum = float(np.sum(t.d.qpos[t.fq] - t.finger_lower))
            if fq_sum < 0.00105:
                return dict(job_id=job_id, subtask=subtask, seed=seed, ok=False, err="not_grasped")
            t.tracker.event_times.pop("insert", None)
            t.tracker.stable_seconds["insert"] = 0.0
            m0 = t.insertion_metrics()
            t.capture = True
            t.full_samples = []
            t.stage = "align_at_mouth"
            if m0["depth_m"] > -0.0035 and m0["lateral_error_m"] > 0.0005:
                tip_c, objr_c = t.object_pose()
                lift_need = 0.0060 + max(0.0, m0["depth_m"])
                t.object_move(tip_c - t.S[:, 2] * lift_need, objr_c, 0.8)
                _recalib(t)
            d_align = 1.6 if m0["lateral_error_m"] < 0.006 else 2.2
            t.object_move(t.port - t.S[:, 2] * 0.004, t.S, d_align)
            _recalib(t)
            t.object_move(t.port - t.S[:, 2] * 0.001, t.S, 0.9)
            _recalib(t)
            t.stage = "insert"
            t.object_move(t.port + t.S[:, 2] * 0.0118, t.S, 3.2)
            t.stage = "verify_insertion"
            t.hold(0.8)
            m1 = t.insertion_metrics()
            if "insert" not in t.tracker.event_times or m1["depth_m"] < 0.010 or len(t.full_samples) < 40:
                return dict(job_id=job_id, subtask=subtask, seed=seed, ok=False, err=f"no_insert_dep{m1['depth_m']*1e3:.1f}")
            for row in t.full_samples:
                row["phase"] = "insert"
            fam = "dagger_p4"
        elif subtask == "p7":
            _recalib(t)
            fq_sum = float(np.sum(t.d.qpos[t.fq] - t.finger_lower))
            if fq_sum < 0.00105:
                return dict(job_id=job_id, subtask=subtask, seed=seed, ok=False, err="not_grasped")
            t.tracker.event_times.pop("return", None)
            t.tracker.stable_seconds["return"] = 0.0
            t.capture = True
            t.full_samples = []
            t.stage = "replace_in_holder"
            tip_c, _ = t.object_pose()
            z_rel = float(tip_c[2] - t.initial_tip[2])
            if z_rel < 0.0185:
                t.object_move(np.array([tip_c[0], tip_c[1], t.initial_tip[2] + 0.021]), t.initial_rotation, 0.8)
                _recalib(t)
            t.object_move(t.initial_tip + np.array([0.0, 0.0, 0.020]), t.initial_rotation, 1.4)
            _recalib(t)
            t.placement_feedback(t.initial_tip + np.array([0.0, 0.0, 0.0175]), t.initial_rotation, 45)
            goals = np.linspace(
                t.initial_tip + np.array([0.0, 0.0, 0.0175]),
                t.initial_tip,
                max(2, int(np.ceil(0.0175 / 0.0007)) + 1),
            )[1:]
            for goal in goals:
                t.placement_feedback(goal, t.initial_rotation, 16)
                if "return" in t.tracker.event_times:
                    break
            if "return" not in t.tracker.event_times:
                t.placement_feedback(t.initial_tip, t.initial_rotation, 60)
            t.hold(0.3)
            if "return" not in t.tracker.event_times or len(t.full_samples) < 40:
                return dict(job_id=job_id, subtask=subtask, seed=seed, ok=False, err="no_return")
            for row in t.full_samples:
                row["phase"] = "replace_in_holder"
            fam = "dagger_p7"
        else:
            return dict(job_id=job_id, subtask=subtask, seed=seed, ok=False, err=f"bad_subtask_{subtask}")

        if t.tracker.invalid_reason is not None:
            return dict(job_id=job_id, subtask=subtask, seed=seed, ok=False, err=f"invalid_{t.tracker.invalid_reason}")

        meta = dict(
            job_id=job_id, family=fam, parent_scene_seed=seed,
            q_home0=item["q_home0"], frames=len(t.full_samples), verified=True,
        )
        _save_surgical_episode(ep_dir, t.full_samples, meta)
        return dict(job_id=job_id, subtask=subtask, seed=seed, ok=True, cached=False, frames=len(t.full_samples))
    except Exception as e:
        return dict(job_id=job_id, subtask=subtask, seed=seed, ok=False, err=f"{type(e).__name__}: {e}"[:150])
    finally:
        if t is not None:
            _close_renderers(t)
        shutil.rmtree(scratch, ignore_errors=True)


def cmd_recover(args: argparse.Namespace) -> None:
    items = []
    for f in sorted(DAGGER_ROOT.glob("harvested_shard*.json")):
        items.extend(json.loads(f.read_text()))
    forbidden = _forbidden_seeds()
    for it in items:
        assert int(it["seed"]) not in forbidden, f"LEAKAGE seed={it['seed']}"
    egl_ids = [int(x.strip()) for x in args.egl_gpu_ids.split(",") if x.strip()] if args.egl_gpu_ids else list(range(args.egl_gpus))
    print(f"[DAGGER RECOVER] Running smooth teacher recovery on {len(items)} real on-policy snapshots (w={args.workers}, egl_gpus={egl_ids})...", flush=True)
    ctx = mp.get_context("spawn")
    jobs = [(i, it, egl_ids) for i, it in enumerate(items)]
    res_all = []
    ok_n = 0
    t0 = time.time()
    with ctx.Pool(processes=args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_recover_one, jobs), 1):
            res_all.append(r)
            if r.get("ok"):
                ok_n += 1
            if i % 50 == 0 or i == len(jobs):
                print(f"  [DAGGER RECOVER {i}/{len(jobs)}] ok={ok_n} ({ok_n/i*100:.1f}%) elapsed={time.time()-t0:.1f}s", flush=True)
    (DAGGER_ROOT / "recover_manifest.json").write_text(json.dumps(res_all, indent=2) + "\n")
    print(f"[DAGGER RECOVER DONE] {ok_n}/{len(jobs)} verified on-policy recovery episodes!", flush=True)


def cmd_merge(args: argparse.Namespace) -> None:
    t0 = time.time()
    forbidden = _forbidden_seeds()
    shutil.rmtree(MERGED_DS, ignore_errors=True)
    (MERGED_DS / "meta").mkdir(parents=True, exist_ok=True)

    base_info = json.loads((BASE_DS / "meta/info.json").read_text())
    features = base_info["features"]
    ref_schema = pq.read_table(BASE_DS / "data/chunk-000/episode_000000.parquet").schema
    chunk_size = 1000

    # Copy base episodes & symlinks cleanly (zero video duplication!)
    base_eps = [json.loads(l) for l in (BASE_DS / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    base_stats = [json.loads(l) for l in (BASE_DS / "meta/episodes_stats.jsonl").read_text().splitlines() if l.strip()]
    base_prov = json.loads((BASE_DS / "source_provenance.json").read_text())

    ep_idx = len(base_eps)
    global_idx = int(base_info["total_frames"])

    for p in sorted((BASE_DS / "data").glob("chunk-*/*.parquet")):
        rel = p.relative_to(BASE_DS)
        dst = MERGED_DS / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(p.resolve()), str(dst))
    for p in sorted((BASE_DS / "videos").glob("chunk-*/*/*.mp4")):
        rel = p.relative_to(BASE_DS)
        dst = MERGED_DS / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(p.resolve()), str(dst))

    episodes = list(base_eps)
    stats = list(base_stats)
    provenance = list(base_prov)

    fam_to_ti = {"dagger_p1": 1, "dagger_p3": 3, "dagger_p4": 4, "dagger_p7": 7}
    dag_dirs = sorted(d for d in (DAGGER_ROOT / "episodes").iterdir() if d.is_dir() and (d / "meta.json").exists())
    added = 0
    added_by_fam = {k: 0 for k in fam_to_ti}
    for repeat in range(max(1, args.dagger_repeat)):
        for ep_dir in dag_dirs:
            m = json.loads((ep_dir / "meta.json").read_text())
            if not m.get("verified"):
                continue
            fam = str(m["family"])
            if fam not in fam_to_ti:
                continue
            seed = int(m["parent_scene_seed"])
            assert seed not in forbidden, f"LEAKAGE seed={seed}"
            vid_srcs = {key: (ep_dir / f"{cam}_lossless.mp4").resolve() for key, cam in CAMERAS.items()}
            if not all(v.exists() for v in vid_srcs.values()):
                continue
            with np.load(ep_dir / "demonstration_full.npz", allow_pickle=True) as d:
                q_home0 = np.asarray(m["q_home0"], dtype=np.float32)
                raw_st = d["observation.state"].astype(np.float32)
                raw_act = d["action"].astype(np.float32)
                raw_ts = d["timestamp"].astype(np.float64)
                ti_val = fam_to_ti[fam]
                n = len(raw_st)
                if n < 16:
                    continue
                state21 = np.ascontiguousarray(usb12.make_state21(raw_st, q_home0), dtype=np.float32)
                raw_act = np.ascontiguousarray(raw_act, dtype=np.float32)
                task_idx = np.full(n, ti_val, dtype=np.int64)
            chunk = ep_idx // chunk_size
            (MERGED_DS / f"data/chunk-{chunk:03d}").mkdir(parents=True, exist_ok=True)
            for key, src in vid_srcs.items():
                dst = MERGED_DS / f"videos/chunk-{chunk:03d}" / key / f"episode_{ep_idx:06d}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(src), str(dst))
            ts_vid = np.arange(0, n, dtype=np.float32) / 30.0
            arr_map = {
                "timestamp": pa.array(ts_vid, type=pa.float32()),
                "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
                "episode_index": pa.array(np.full(n, ep_idx, dtype=np.int64), type=pa.int64()),
                "index": pa.array(np.arange(global_idx, global_idx + n, dtype=np.int64), type=pa.int64()),
                "task_index": pa.array(task_idx, type=pa.int64()),
                "state": pa.FixedSizeListArray.from_arrays(pa.array(state21.reshape(-1), type=pa.float32()), 21),
                "actions": pa.FixedSizeListArray.from_arrays(pa.array(raw_act.reshape(-1), type=pa.float32()), 8),
                "source_frame": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
                "simulation_timestamp": pa.array(raw_ts, type=pa.float64()),
            }
            tbl_out = pa.Table.from_arrays([arr_map[f.name] for f in ref_schema], schema=ref_schema)
            pq.write_table(tbl_out, str(MERGED_DS / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"))
            episodes.append(dict(episode_index=ep_idx, tasks=[usb12.PHASE_PROMPTS[ti_val]], length=n))
            ep_stats = dict(
                state=numeric_stats(state21),
                actions=numeric_stats(raw_act),
                timestamp=numeric_stats((np.arange(0, n, dtype=np.float64) / 30.0)[:, None]),
                frame_index=numeric_stats(np.arange(n, dtype=np.float64)[:, None]),
                episode_index=numeric_stats(np.full((n, 1), ep_idx, dtype=np.float64)),
                index=numeric_stats(np.arange(global_idx, global_idx + n, dtype=np.float64)[:, None]),
                task_index=numeric_stats(task_idx.astype(np.float64)[:, None]),
                source_frame=numeric_stats(np.arange(0, n, dtype=np.float64)[:, None]),
                simulation_timestamp=numeric_stats(raw_ts[:, None]),
            )
            stats.append(dict(episode_index=ep_idx, stats=ep_stats))
            provenance.append(dict(episode_index=ep_idx, length=n, source="onpolicy_dagger", family=m["family"], parent_scene_seed=seed, repeat=repeat))
            global_idx += n
            ep_idx += 1
            added += 1
            added_by_fam[fam] += 1

    info = create_empty_dataset_info("v2.1", 30, "g1_usb_right_arm_rel256_phase12", features, True)
    info.update(
        total_episodes=ep_idx,
        total_frames=global_idx,
        total_tasks=len(usb12.PHASE_PROMPTS),
        total_videos=ep_idx * len(CAMERAS),
        total_chunks=(ep_idx + chunk_size - 1) // chunk_size,
        chunks_size=chunk_size,
        splits={"train": f"0:{ep_idx}"},
    )
    (MERGED_DS / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (MERGED_DS / "meta/tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": p}) + "\n" for i, p in enumerate(usb12.PHASE_PROMPTS))
    )
    (MERGED_DS / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (MERGED_DS / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats))
    (MERGED_DS / "source_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[MERGED DATASET READY] {MERGED_DS}: base={len(base_eps)} + dagger={added} ({added_by_fam}, x{args.dagger_repeat}) -> total={ep_idx} eps, {global_idx} frames in {time.time()-t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("rollout")
    p1.add_argument("--config-dir", required=True)
    p1.add_argument("--config-name", required=True)
    p1.add_argument("--checkpoint", required=True)
    p1.add_argument("--limit", type=int, default=400)
    p1.add_argument("--max-steps", type=int, default=2600)
    p1.add_argument("--shard-idx", type=int, default=0)
    p1.add_argument("--num-shards", type=int, default=4)

    p2 = sub.add_parser("recover")
    p2.add_argument("--workers", type=int, default=48)
    p2.add_argument("--egl-gpus", type=int, default=4)
    p2.add_argument("--egl-gpu-ids", default="")

    p3 = sub.add_parser("merge")
    p3.add_argument("--dagger-repeat", type=int, default=2)

    args = ap.parse_args()
    if args.cmd == "rollout":
        cmd_rollout(args)
    elif args.cmd == "recover":
        cmd_recover(args)
    else:
        cmd_merge(args)


if __name__ == "__main__":
    main()
