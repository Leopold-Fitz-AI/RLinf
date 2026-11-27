# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 Clean NVMe-Native High-Density Data Factory for scenes_v3_real (from-pi05_base SFT).

Fixes ALL 5 forensic root causes from prior runs:
  1. EXACT speedup=1.5 (matching EXP12 dynamics: P1~86 frames, P1..P9~1,370 frames, |act-q| p95~0.0055 rad).
  2. EXACT grasp depth=+0.008 m (z_grasp=0.7985 dead-center on 19mm overmold, yaw=90, ELLIPTIC, impratio=10).
  3. CleanSmoothP4Teacher quintic moves (zero 60-tick mid-air stalls) + dynamic in-hand (offset, rel) recalibration.
  4. Full 1,200 high-density superset train scenes (±13mm, ±6deg) + 3-tier P4 recovery (tight/core/wide up to ±16mm)
     + 3-tier P7 holder recovery (lip/core/wide up to ±10mm) + unimodal P0/P1/P9/P10/P11 transit & brake.
  5. Recomputes exact fresh norm_stats_v3_fresh.json on 21D state (symmetric [-0.25,+0.25] home-delta) and
     clamped 16-step relative actions from the new dataset itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/vla-cd/jy/v12b_usb_cycle_v1")
EXP10 = ROOT / "clean_rel256_phase10_v1"
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

import datasets
import mujoco
import numpy as np
from lerobot.common.datasets.utils import (
    DEFAULT_FEATURES,
    create_empty_dataset_info,
    get_hf_features_from_features,
)
from scipy.spatial.transform import Rotation, Slerp

from g1_usb_sim.lossless_episode import encode_rgb
from grasp_stability_final_teacher import VerticalWideForceTeacher
import usb_rel256_phase12 as usb12

NVME_ROOT = Path("/mlplatform/jy/v3_fresh_nvme")
SCENES_MANIFEST = NVME_ROOT / "train_scenes_v3_1200.json"
RAW_FULL = NVME_ROOT / "raw_full_1200"
RAW_SURG = NVME_ROOT / "raw_surgical"
DS_OUT = NVME_ROOT / "train_v3_fresh"
NORM_OUT_NVME = NVME_ROOT / "norm_stats_v3_fresh.json"
NORM_OUT_DURABLE = H2 / "assets/norm_stats_v3_fresh.json"

YAW_SEARCH_JSON = ROOT / "comparisons/cw90_wide_grasp_v1/vertical_yaw_search.json"
WAYPOINTS_JSON = EXP10 / "assets/canonical_waypoints.json"
HOME_BRAKE_THRESH = 0.16
GRASP_DEPTH_M = 0.008
MOTION_SPEEDUP = 1.5
HORIZON = 16

CAMERAS = {
    "image": "front",
    "left_wrist_image": "left_wrist",
    "wrist_image": "wrist",
}
REQ_EVENTS = {"lift", "insert", "withdraw", "return", "release", "home"}


class CleanSmoothTeacher(VerticalWideForceTeacher):
    """Exact VerticalWideForceTeacher with smooth quintic moves and compact gravity-sag trim (no 60-tick freeze stalls)."""

    def move(self, p, r, duration=2.0, settle=0.12):
        p = np.asarray(p, dtype=float)
        r = np.asarray(r, dtype=float)
        p0, r0 = self.pose()
        rotmag = float(np.linalg.norm(Rotation.from_matrix(r @ r0.T).as_rotvec()))
        speed = min(self.speedup, 1.5) if self.stage == "insert" else self.speedup
        dur = max(float(duration), float(np.linalg.norm(p - p0)) / 0.035 * 1.9, rotmag / 0.3 * 1.9) / speed
        count = max(1, int(np.ceil(dur / self.control_dt)))
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix([r0, r]))
        for i in range(count):
            u = (i + 1) / count
            s = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
            q = self.ik(p0 + (p - p0) * s, slerp(s).as_matrix())
            delta = q - self.target[self.aq]
            if np.max(np.abs(delta)) > 0.12:
                raise RuntimeError(f"IK branch jump {self.stage}: {np.max(np.abs(delta)):.4f}")
            self.target[self.aq] = q
            self.tick()
        desired = self.ik(p, r)
        for _ in range(6):
            actual, rot = self.pose()
            ep = float(np.linalg.norm(actual - p))
            er = float(np.linalg.norm(Rotation.from_matrix(r @ rot.T).as_rotvec()))
            if ep < 0.00035 and er < 0.008:
                break
            self.target[self.aq] = np.clip(
                self.target[self.aq] + np.clip(desired - self.d.qpos[self.aq], -0.003, 0.003),
                self.lo + 0.001, self.hi - 0.001,
            )
            self.tick()
        if settle > 0.0:
            self.hold(min(float(settle), 0.15))

    def object_move(self, p, r, duration=2.0):
        p = np.asarray(p, dtype=float)
        r = np.asarray(r, dtype=float)
        self.move(p + r @ self.offset, r @ self.rel, duration=duration, settle=0.12)


def _load_pregrasp_seed_yaw90() -> np.ndarray:
    rows = json.loads(YAW_SEARCH_JSON.read_text())
    return np.asarray(next(v["q"] for v in rows if v["yaw_deg"] == 90), dtype=np.float64)


def _load_waypoints() -> list[np.ndarray]:
    wp = json.loads(WAYPOINTS_JSON.read_text())["waypoints"]
    return [np.asarray(w, dtype=np.float64) for w in wp]


def _close_renderers(t) -> None:
    for obj in (getattr(t, "renderer", None), getattr(t, "observer_renderer", None)):
        if obj is not None and hasattr(obj, "close"):
            try:
                obj.close()
            except Exception:
                pass


def _recalib(t) -> None:
    """Re-measure live in-hand (offset, rel) so object_move tracks actual USB tip under pure friction."""
    tip_c, objr_c = t.object_pose()
    p_c, r_c = t.pose()
    t.offset = objr_c.T @ (p_c - tip_c)
    t.rel = objr_c.T @ r_c


def make_teacher(
    scene_path: Path,
    scratch_dir: Path,
    seed: int,
    brightness: float,
    *,
    capture: bool = False,
    storage: str = "lossless-video",
    speedup: float = MOTION_SPEEDUP,
    smooth: bool = False,
):
    shutil.rmtree(scratch_dir, ignore_errors=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    cls = CleanSmoothTeacher if smooth else VerticalWideForceTeacher
    t = cls(
        scene_path, scratch_dir, seed,
        jitter=0.0, closed=-0.020, face="wide", tilt=0.0,
        capture=capture, brightness=brightness, storage=storage,
    )
    t.rel = t.rel @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
    t.offset = np.array([0.0, 0.0, -0.025]) + t.rel[:, 2] * 0.025
    t.offset[2] += GRASP_DEPTH_M
    t.pickup_clearance = 0.03
    t.face = "wide"
    t.speedup = float(speedup)
    t.m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    t.m.opt.impratio = 10
    t.pregrasp_seed = _load_pregrasp_seed_yaw90()
    return t


def numeric_stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return dict(
        min=a.min(axis=0).tolist(),
        max=a.max(axis=0).tolist(),
        mean=a.mean(axis=0).tolist(),
        std=a.std(axis=0).tolist(),
        count=[int(a.shape[0])],
    )


def _collect_full_one(args: tuple[int, dict, int]) -> dict:
    w_idx, job, n_egl_gpus = args
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(w_idx % max(1, n_egl_gpus))
    seed = int(job["seed"])
    scene_p = Path(job["scene"])
    brightness = float(job["brightness"])
    ep_dir = RAW_FULL / str(seed)
    done_marker = ep_dir / "v3f_done.json"

    if done_marker.exists() and (ep_dir / "demonstration_full.npz").exists() and (ep_dir / "front_lossless.mp4").exists():
        meta = json.loads(done_marker.read_text())
        if meta.get("ok"):
            return meta

    tmp_dir = Path("/dev/shm/jy_h2_tmp") / f"col_v3f_full_{os.getpid()}_{seed}"
    t = None
    try:
        t = make_teacher(scene_p, tmp_dir, seed, brightness, capture=True, storage="lossless-video", speedup=MOTION_SPEEDUP, smooth=False)
        final_rep = t.run()
        t.save(True, final=final_rep)
        rep_p = tmp_dir / "result.json"
        if rep_p.exists():
            rep = json.loads(rep_p.read_text())
            phys = rep.get("final") or rep.get("outcomes") or rep
            ev = set((phys.get("event_times") or {}).keys())
            inv = phys.get("invalid_reason")
        else:
            ev = set(getattr(t.tracker, "event_times", {}).keys()) if getattr(t, "tracker", None) else set()
            inv = None
        req_files = [
            "demonstration_full.npz", "front_lossless.mp4", "wrist_lossless.mp4", "left_wrist_lossless.mp4",
            "cycle_phase_raise_to_pregrasp.npz", "cycle_phase_approach_holder.npz",
            "cycle_phase_align_at_mouth.npz", "cycle_phase_replace_in_holder.npz",
            "cycle_phase_clear_holder.npz", "cycle_phase_final_hold.npz",
        ]
        ok = REQ_EVENTS.issubset(ev) and (inv is None) and all((tmp_dir / f).exists() for f in req_files)
        if ok:
            tmp_dst = ep_dir.with_name(ep_dir.name + ".tmp")
            shutil.rmtree(tmp_dst, ignore_errors=True)
            shutil.copytree(tmp_dir, tmp_dst)
            if ep_dir.exists():
                shutil.rmtree(ep_dir)
            os.replace(tmp_dst, ep_dir)
            meta = dict(seed=seed, ok=True, events=sorted(ev), invalid=inv, path=str(ep_dir))
            done_marker.write_text(json.dumps(meta, indent=2) + "\n")
            return meta
        return dict(seed=seed, ok=False, events=sorted(ev), invalid=inv, err="incomplete_or_missing_files")
    except Exception as e:
        return dict(seed=seed, ok=False, events=[], invalid="exception", err=f"{type(e).__name__}: {e}"[:180])
    finally:
        if t is not None:
            _close_renderers(t)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _save_surgical_episode(ep_dir: Path, rows: list[dict], meta: dict) -> None:
    tmp_ep = ep_dir.with_name(ep_dir.name + ".tmp")
    shutil.rmtree(tmp_ep, ignore_errors=True)
    tmp_ep.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for cam in ("front", "wrist", "left_wrist"):
        key = "observation.images." + cam
        vid_name = cam + "_lossless.mp4"
        manifest[key] = encode_rgb(tmp_ep / vid_name, np.asarray([r[key] for r in rows]))
    np.savez_compressed(
        tmp_ep / "demonstration_full.npz",
        **{k: np.asarray([r[k] for r in rows]) for k in rows[0] if not k.startswith("observation.images.")},
    )
    (tmp_ep / "video_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (tmp_ep / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if ep_dir.exists():
        shutil.rmtree(ep_dir)
    os.replace(tmp_ep, ep_dir)


def _collect_surgical_one(args: tuple[int, dict, int]) -> dict:
    w_idx, job, n_egl_gpus = args
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(w_idx % max(1, n_egl_gpus))
    ep_id = job["job_id"]
    fam = job["family"]
    seed = int(job["seed"])
    sc = job["scene"]
    rec_dir = Path(job["rec_path"])
    ep_dir = RAW_SURG / ep_id

    if (ep_dir / "meta.json").exists() and (ep_dir / "demonstration_full.npz").exists() and (ep_dir / "front_lossless.mp4").exists():
        m = json.loads((ep_dir / "meta.json").read_text())
        if m.get("verified"):
            return dict(job_id=ep_id, family=fam, seed=seed, ok=True, cached=True, frames=int(m.get("frames", 0)))

    rng = np.random.default_rng(1800000 + int(job["idx"]) * 73 + abs(hash(fam)) % 10000)
    with np.load(rec_dir / "demonstration_full.npz", allow_pickle=True) as d:
        q_home0 = d["observation.state"][0, 6:13].astype(np.float32).copy()
        ph_arr = np.array([str(x) for x in d["phase"]])
        app_ids = np.where(ph_arr == "approach_holder")[0]
        q_pregrasp = d["observation.state"][int(app_ids[0]), 6:13].astype(np.float32).copy()

    waypoints = _load_waypoints()
    scratch = Path("/dev/shm/jy_h2_tmp") / f"col_v3f_surg_{os.getpid()}_{ep_id}"
    t = None
    try:
        use_smooth = fam.startswith("p4_rec_") or fam.startswith("p7_rec_")
        t = make_teacher(
            Path(sc["scene"]), scratch, seed, float(sc["brightness"]),
            capture=False, speedup=MOTION_SPEEDUP, smooth=use_smooth,
        )

        if fam == "unimodal_raise_approach":
            t.restore(rec_dir / "cycle_phase_raise_to_pregrasp.npz")
            t.stage = "raise_to_pregrasp"
            t.speedup = MOTION_SPEEDUP
            t.capture = True
            t.full_samples = []
            for qw in waypoints:
                t.joints(qw, 1.0)
            t.joints(q_pregrasp, 1.0)
            t.stage = "approach_holder"
            p_pre, r_pre = t.pose()
            p_goal = p_pre - np.array([0.0, 0.0, 0.030])
            t.move(p_goal, r_pre, 3.0, settle=0.20)
            end_p, _ = t.pose()
            err_mm = float(np.linalg.norm(end_p - p_goal) * 1000.0)
            if err_mm > 1.5 or len(t.full_samples) < 40:
                return dict(job_id=ep_id, family=fam, seed=seed, ok=False, err=f"approach_err={err_mm:.2f}mm")
            info = dict(err_mm=round(err_mm, 3))

        elif fam == "approach_recovery":
            t.restore(rec_dir / "cycle_phase_approach_holder.npz")
            snap_p, snap_r = t.pose()
            p_goal = snap_p - np.array([0.0, 0.0, 0.030])
            t.stage = "approach_holder"
            dx = float(rng.uniform(-0.025, 0.025))
            dy = float(rng.uniform(-0.025, 0.025))
            dz = float(rng.uniform(-0.006, 0.025))
            t.move(snap_p + np.array([dx, dy, dz]), snap_r, 1.0, settle=0.08)
            t.capture = True
            t.full_samples = []
            above = p_goal + np.array([0.0, 0.0, max(0.015, 0.030 + min(dz, 0.0))])
            t.move(above, snap_r, 1.8, settle=0.05)
            t.move(p_goal, snap_r, 2.5, settle=0.20)
            end_p, _ = t.pose()
            err_mm = float(np.linalg.norm(end_p - p_goal) * 1000.0)
            if err_mm > 1.0 or len(t.full_samples) < 25:
                return dict(job_id=ep_id, family=fam, seed=seed, ok=False, err=f"err={err_mm:.2f}mm")
            info = dict(pert_mm=[round(dx * 1e3, 1), round(dy * 1e3, 1), round(dz * 1e3, 1)], err_mm=round(err_mm, 3))

        elif fam in ("p4_rec_tight", "p4_rec_core", "p4_rec_wide"):
            t.restore(rec_dir / "cycle_phase_align_at_mouth.npz")
            t.d.eq_active[t.weld] = False
            _recalib(t)
            t.stage = "align_at_mouth"
            if fam == "p4_rec_tight":
                dx = float(rng.uniform(-0.0022, 0.0022))
                dy = float(rng.uniform(-0.0022, 0.0022))
                dz = float(rng.uniform(0.0005, 0.0040))
                rl = float(rng.uniform(-1.2, 1.2))
                pt = float(rng.uniform(-1.2, 1.2))
                yw = float(rng.uniform(-3.5, 3.5))
            elif fam == "p4_rec_core":
                dx = float(rng.uniform(-0.0080, 0.0080))
                dy = float(rng.uniform(-0.0080, 0.0080))
                dz = float(rng.uniform(0.0040, 0.0200))
                rl = float(rng.uniform(-2.2, 2.2))
                pt = float(rng.uniform(-2.2, 2.2))
                yw = float(rng.uniform(-8.0, 8.0))
            else:
                ang = float(rng.uniform(0.0, 2.0 * np.pi))
                rad = float(rng.uniform(0.0080, 0.0160))
                dx, dy = rad * math.cos(ang), rad * math.sin(ang)
                dz = float(rng.uniform(0.0080, 0.0220))
                rl = float(rng.uniform(-2.0, 2.0))
                pt = float(rng.uniform(-2.0, 2.0))
                yw = float(rng.uniform(-8.0, 8.0))

            r_pert = t.S @ Rotation.from_euler("xyz", [rl, pt, yw], degrees=True).as_matrix()
            p_pert = t.port + t.S @ np.array([dx, dy, -dz])
            t.object_move(p_pert, r_pert, 1.2)
            _recalib(t)

            fq_sum = float(np.sum(t.d.qpos[t.fq] - t.finger_lower))
            if fq_sum < 0.00105:
                return dict(job_id=ep_id, family=fam, seed=seed, ok=False, err="dropped_pre")

            t.tracker.event_times.pop("insert", None)
            t.tracker.stable_seconds["insert"] = 0.0
            m0 = t.insertion_metrics()

            t.capture = True
            t.full_samples = []
            t.stage = "align_at_mouth"
            if m0["depth_m"] > -0.0030 and m0["lateral_error_m"] > 0.0005:
                tip_c, objr_c = t.object_pose()
                lift_need = 0.0050 + m0["depth_m"]
                t.object_move(tip_c - t.S[:, 2] * lift_need, objr_c, 0.7)
                _recalib(t)

            d_align = 1.6 if m0["lateral_error_m"] < 0.006 else 2.1
            t.object_move(t.port - t.S[:, 2] * 0.004, t.S, d_align)
            _recalib(t)
            t.object_move(t.port - t.S[:, 2] * 0.001, t.S, 0.9)
            _recalib(t)

            t.stage = "insert"
            t.object_move(t.port + t.S[:, 2] * 0.0118, t.S, 3.2)
            t.stage = "verify_insertion"
            t.hold(0.8)

            m1 = t.insertion_metrics()
            if "insert" not in t.tracker.event_times or m1["depth_m"] < 0.010 or len(t.full_samples) < 50:
                return dict(job_id=ep_id, family=fam, seed=seed, ok=False, err=f"no_insert_dep{m1['depth_m']*1e3:.1f}")
            for row in t.full_samples:
                row["phase"] = "insert"
            info = dict(
                init_lat_mm=round(m0["lateral_error_m"] * 1e3, 2),
                init_dep_mm=round(m0["depth_m"] * 1e3, 2),
                init_ang_deg=round(float(np.degrees(m0["angle_error_rad"])), 2),
                final_dep_mm=round(m1["depth_m"] * 1e3, 2),
            )

        elif fam in ("p7_rec_lip", "p7_rec_core", "p7_rec_wide"):
            t.restore(rec_dir / "cycle_phase_replace_in_holder.npz")
            t.d.eq_active[t.weld] = False
            _recalib(t)
            t.stage = "replace_in_holder"
            if fam == "p7_rec_lip":
                ang = float(rng.uniform(0.0, 2.0 * np.pi))
                rad = float(rng.uniform(0.0003, 0.0015))
                dx, dy = rad * math.cos(ang), rad * math.sin(ang)
                z_above = float(rng.uniform(0.0168, 0.0205))
                yw = float(rng.uniform(-2.5, 2.5))
            elif fam == "p7_rec_core":
                dx = float(rng.uniform(-0.0060, 0.0060))
                dy = float(rng.uniform(-0.0060, 0.0060))
                z_above = float(rng.uniform(0.0185, 0.0280))
                yw = float(rng.uniform(-6.0, 6.0))
            else:
                dx = float(rng.uniform(-0.0100, 0.0100))
                dy = float(rng.uniform(-0.0100, 0.0100))
                z_above = float(rng.uniform(0.0200, 0.0300))
                yw = float(rng.uniform(-6.0, 6.0))

            r_pert = t.initial_rotation @ Rotation.from_euler("z", yw, degrees=True).as_matrix()
            p_above_hi = t.initial_tip + np.array([dx, dy, max(z_above, 0.026)])
            p_start = t.initial_tip + np.array([dx, dy, z_above])
            t.object_move(p_above_hi, r_pert, 0.9)
            t.object_move(p_start, r_pert, 0.9)
            _recalib(t)

            fq_sum = float(np.sum(t.d.qpos[t.fq] - t.finger_lower))
            if fq_sum < 0.00105:
                return dict(job_id=ep_id, family=fam, seed=seed, ok=False, err="dropped_pre")

            t.tracker.event_times.pop("return", None)
            t.tracker.stable_seconds["return"] = 0.0
            t.capture = True
            t.full_samples = []

            start_z = max(z_above, 0.0185)
            # First smoothly align above slot opening, then feedback-descend into white 4.8x12.4mm slot
            t.object_move(t.initial_tip + np.array([0.0, 0.0, start_z]), t.initial_rotation, 1.4 if fam != "p7_rec_lip" else 1.0)
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
                return dict(job_id=ep_id, family=fam, seed=seed, ok=False, err="no_return")
            for row in t.full_samples:
                row["phase"] = "replace_in_holder"
            info = dict(pert_mm=[round(dx * 1e3, 2), round(dy * 1e3, 2), round(z_above * 1e3, 1)], yaw_deg=round(yw, 2))

        elif fam == "unimodal_clear_home":
            t.restore(rec_dir / "cycle_phase_clear_holder.npz")
            t.stage = "clear_holder"
            t.speedup = MOTION_SPEEDUP
            t.capture = True
            t.full_samples = []
            pc, rc = t.pose()
            t.move(pc + np.array([0.0, 0.0, t.pickup_clearance]), rc, 3.0)
            t.stage = "return_to_side"
            for qw in reversed(waypoints):
                t.joints(qw, 1.0)
            t.joints(t.home_q[t.aq], 1.0)
            t.stage = "restore_fingers"
            t.target[t.fq] = t.home_q[t.fq]
            t.hold(0.5)
            t.stage = "final_hold"
            t.hold(1.0)
            if "home" not in t.tracker.event_times:
                return dict(job_id=ep_id, family=fam, seed=seed, ok=False, err="no_home")
            info = dict(events=sorted(t.tracker.event_times.keys()))

        elif fam == "symmetric_home_brake_p11":
            t.restore(rec_dir / "cycle_phase_final_hold.npz")
            t.tracker.event_times.pop("home", None)
            t.tracker.stable_seconds["home"] = 0.0
            t.stage = "final_hold"
            mag = 0.045 if (int(job["idx"]) % 2 == 0) else 0.16
            delta_q = rng.uniform(-mag, mag, size=7)
            if np.max(np.abs(delta_q)) < 0.008:
                delta_q[int(job["idx"]) % 7] = (1.0 if (int(job["idx"]) % 4 < 2) else -1.0) * 0.02
            q_pert = np.clip(t.home_q[t.aq] + delta_q, t.lo + 0.005, t.hi - 0.005)
            t.speedup = 1.5
            t.joints(q_pert, 0.7)
            t.capture = True
            t.full_samples = []
            t.speedup = 1.2
            t.joints(t.home_q[t.aq], 0.65 if mag < 0.08 else 1.1)
            t.hold(1.2)
            if "home" not in t.tracker.event_times:
                return dict(job_id=ep_id, family=fam, seed=seed, ok=False, err="no_home")
            for row in t.full_samples:
                row["phase"] = "final_home_brake"
            info = dict(mag_rad=mag, pert_max=round(float(np.max(np.abs(delta_q))), 4))
        else:
            raise ValueError(fam)

        if t.tracker.invalid_reason is not None:
            return dict(job_id=ep_id, family=fam, seed=seed, ok=False, err=f"invalid={t.tracker.invalid_reason}")

        meta = dict(
            job_id=ep_id, family=fam, parent_scene_seed=seed,
            source_record_path=str(rec_dir), frames=len(t.full_samples),
            q_home0=q_home0.tolist(), verified=True, info=info,
        )
        _save_surgical_episode(ep_dir, t.full_samples, meta)
        return dict(job_id=ep_id, family=fam, seed=seed, ok=True, cached=False, frames=len(t.full_samples))
    except Exception as e:
        return dict(job_id=ep_id, family=fam, seed=seed, ok=False, err=f"{type(e).__name__}: {e}"[:180])
    finally:
        if t is not None:
            _close_renderers(t)
        shutil.rmtree(scratch, ignore_errors=True)


def compute_episode_rel_chunks(state21: np.ndarray, actions8: np.ndarray) -> np.ndarray:
    """Exact clamped 16-step relative action chunks (N*16, 8) matching USBRel256Phase12Inputs."""
    n = len(state21)
    offsets = np.arange(HORIZON)[None, :]
    base_idx = np.arange(n)[:, None]
    idx = np.minimum(base_idx + offsets, n - 1)
    chunks = actions8[idx].copy()
    chunks[:, :, :7] -= state21[:, None, usb12.RIGHT_Q_SLICE]
    return chunks.reshape(-1, 8)


def build_lerobot_and_norm_stats(ok_full: list[dict], forbidden_seeds: set[int]) -> None:
    t0 = time.time()
    shutil.rmtree(DS_OUT, ignore_errors=True)
    (DS_OUT / "meta").mkdir(parents=True, exist_ok=True)
    (DS_OUT / "data/chunk-000").mkdir(parents=True, exist_ok=True)

    features = {
        **DEFAULT_FEATURES,
        "state": {"dtype": "float32", "shape": (21,), "names": None},
        "actions": {"dtype": "float32", "shape": (8,), "names": None},
        "source_frame": {"dtype": "int64", "shape": (1,), "names": None},
        "simulation_timestamp": {"dtype": "float64", "shape": (1,), "names": None},
    }
    for key in CAMERAS:
        features[key] = {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": 30, "video.height": 224, "video.width": 224,
                "video.codec": "h264", "video.pix_fmt": "gbrp",
                "video.is_depth_map": False, "has_audio": False,
            },
        }
    hf_features = get_hf_features_from_features(
        {k: v for k, v in features.items() if v["dtype"] not in ("image", "video")}
    )

    chunk_size = 1000
    episodes, stats, provenance = [], [], []
    task_frame_counts = [0] * len(usb12.PHASE_PROMPTS)
    ep_idx, global_idx = 0, 0
    all_states: list[np.ndarray] = []
    all_rel_actions: list[np.ndarray] = []

    def emit(state21, action8, task_idx, ts_sim, lo, hi, vid_srcs, prov):
        nonlocal ep_idx, global_idx
        n = len(state21)
        if n < 16:
            return
        assert np.isfinite(state21).all() and np.isfinite(action8).all(), f"Non-finite in {prov}"
        for ti_v in task_idx:
            task_frame_counts[int(ti_v)] += 1
        all_states.append(state21.astype(np.float32))
        all_rel_actions.append(compute_episode_rel_chunks(state21, action8).astype(np.float32))

        chunk = ep_idx // chunk_size
        (DS_OUT / f"data/chunk-{chunk:03d}").mkdir(parents=True, exist_ok=True)
        for key, src in vid_srcs.items():
            dst = DS_OUT / f"videos/chunk-{chunk:03d}" / key / f"episode_{ep_idx:06d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(src), str(dst))
        ts_vid = np.arange(lo, hi, dtype=np.float32) / 30.0
        table = {
            "timestamp": ts_vid.tolist(),
            "frame_index": list(range(n)),
            "episode_index": [ep_idx] * n,
            "index": list(range(global_idx, global_idx + n)),
            "task_index": task_idx.tolist(),
            "state": state21.tolist(),
            "actions": action8.tolist(),
            "source_frame": list(range(lo, hi)),
            "simulation_timestamp": ts_sim.tolist(),
        }
        datasets.Dataset.from_dict(table, features=hf_features).to_parquet(
            str(DS_OUT / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet")
        )
        unique_tasks = [usb12.PHASE_PROMPTS[int(ti)] for ti in sorted(set(task_idx.tolist()))]
        episodes.append(dict(episode_index=ep_idx, tasks=unique_tasks, length=n))
        ep_stats = dict(
            state=numeric_stats(state21),
            actions=numeric_stats(action8),
            timestamp=numeric_stats((np.arange(lo, hi, dtype=np.float64) / 30.0)[:, None]),
            frame_index=numeric_stats(np.arange(n, dtype=np.float64)[:, None]),
            episode_index=numeric_stats(np.full((n, 1), ep_idx, dtype=np.float64)),
            index=numeric_stats(np.arange(global_idx, global_idx + n, dtype=np.float64)[:, None]),
            task_index=numeric_stats(task_idx.astype(np.float64)[:, None]),
            source_frame=numeric_stats(np.arange(lo, hi, dtype=np.float64)[:, None]),
            simulation_timestamp=numeric_stats(ts_sim[:, None]),
        )
        stats.append(dict(episode_index=ep_idx, stats=ep_stats))
        provenance.append(dict(episode_index=ep_idx, length=n, lo=lo, hi=hi, **prov))
        global_idx += n
        ep_idx += 1

    # 1. Full cycles (P1..P9) + 0.5x clean P4 & P7 anchor upsampling
    for idx_f, rec in enumerate(sorted(ok_full, key=lambda r: int(r["seed"]))):
        seed = int(rec["seed"])
        assert seed not in forbidden_seeds, f"LEAKAGE seed={seed}"
        p = Path(rec["path"])
        vid_srcs = {key: (p / f"{cam}_lossless.mp4").resolve() for key, cam in CAMERAS.items()}
        assert all(v.exists() for v in vid_srcs.values())
        with np.load(p / "demonstration_full.npz", allow_pickle=True) as d:
            phases = np.array([str(s) for s in d["phase"]])
            raw_st = d["observation.state"].astype(np.float32)
            raw_act = d["action"].astype(np.float32)
            raw_ts = d["timestamp"].astype(np.float64)
            q_home0 = raw_st[0, 6:13].copy()

        app_ids = np.where(phases == "approach_holder")[0]
        clr_ids = np.where(phases == "clear_holder")[0]
        lo, hi = int(app_ids[0]), int(clr_ids[-1]) + 1
        sub_phases = phases[lo:hi]
        task_idx = np.array([usb12.FINE_PHASE_TO_TASK_INDEX[ph] for ph in sub_phases], dtype=np.int64)
        emit(
            usb12.make_state21(raw_st[lo:hi], q_home0), raw_act[lo:hi], task_idx, raw_ts[lo:hi], lo, hi, vid_srcs,
            dict(source="v3_fresh_p1_to_p9", parent_scene_seed=seed),
        )
        if idx_f % 2 == 0:
            p4_ids = np.where(np.isin(phases, ["align_at_mouth", "insert", "verify_insertion"]))[0]
            if len(p4_ids) >= 32:
                a, b = int(p4_ids[0]), int(p4_ids[-1]) + 1
                emit(
                    usb12.make_state21(raw_st[a:b], q_home0), raw_act[a:b],
                    np.full(b - a, 4, dtype=np.int64), raw_ts[a:b], a, b, vid_srcs,
                    dict(source="v3_fresh_p4_anchor", parent_scene_seed=seed),
                )
            p7_ids = np.where(phases == "replace_in_holder")[0]
            if len(p7_ids) >= 24:
                a, b = int(p7_ids[0]), int(p7_ids[-1]) + 1
                emit(
                    usb12.make_state21(raw_st[a:b], q_home0), raw_act[a:b],
                    np.full(b - a, 7, dtype=np.int64), raw_ts[a:b], a, b, vid_srcs,
                    dict(source="v3_fresh_p7_anchor", parent_scene_seed=seed),
                )

    # 2. All surgical + 3-tier P4/P7 recovery episodes
    surg_count = 0
    if RAW_SURG.exists():
        for ep_dir in sorted(RAW_SURG.iterdir()):
            mp_file = ep_dir / "meta.json"
            if not (ep_dir.is_dir() and mp_file.exists() and (ep_dir / "demonstration_full.npz").exists()):
                continue
            m = json.loads(mp_file.read_text())
            if not m.get("verified"):
                continue
            seed = int(m["parent_scene_seed"])
            assert seed not in forbidden_seeds, f"LEAKAGE in surgical {ep_dir.name}"
            fam = str(m["family"])
            vid_srcs = {key: (ep_dir / f"{cam}_lossless.mp4").resolve() for key, cam in CAMERAS.items()}
            if not all(v.exists() for v in vid_srcs.values()):
                continue
            with np.load(ep_dir / "demonstration_full.npz", allow_pickle=True) as d:
                q_home0 = np.asarray(m["q_home0"], dtype=np.float32)
                raw_st = d["observation.state"].astype(np.float32)
                raw_act = d["action"].astype(np.float32)
                raw_ts = d["timestamp"].astype(np.float64)
                phases = [str(s) for s in d["phase"]]
                task_idx = np.array([usb12.FINE_PHASE_TO_TASK_INDEX[s] for s in phases], dtype=np.int64)
                if fam == "symmetric_home_brake_p11":
                    task_idx[:] = 11
                elif fam == "unimodal_clear_home":
                    herr = np.max(np.abs(raw_st[:, 6:13] - q_home0[None, :]), axis=1)
                    in_p10 = (task_idx == 10) & (herr <= HOME_BRAKE_THRESH)
                    if np.any(in_p10):
                        task_idx[int(np.where(in_p10)[0][0]):] = 11
                elif fam.startswith("p4_rec_"):
                    task_idx[:] = 4
                elif fam.startswith("p7_rec_"):
                    task_idx[:] = 7
                elif fam == "approach_recovery":
                    task_idx[:] = 1
            emit(
                usb12.make_state21(raw_st, q_home0), raw_act, task_idx, raw_ts, 0, len(raw_st), vid_srcs,
                dict(source="v3_fresh_surgical", family=fam, job_id=m["job_id"], parent_scene_seed=seed),
            )
            surg_count += 1

    info = create_empty_dataset_info(
        codebase_version="v2.1", fps=30, robot_type="g1_usb_right_arm_rel256_phase12",
        features=features, use_videos=True,
    )
    info["total_episodes"] = ep_idx
    info["total_frames"] = global_idx
    info["total_tasks"] = len(usb12.PHASE_PROMPTS)
    info["total_videos"] = ep_idx * len(CAMERAS)
    info["total_chunks"] = (ep_idx + chunk_size - 1) // chunk_size
    info["chunks_size"] = chunk_size
    info["splits"] = {"train": f"0:{ep_idx}"}
    (DS_OUT / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (DS_OUT / "meta/tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": p}) + "\n" for i, p in enumerate(usb12.PHASE_PROMPTS))
    )
    (DS_OUT / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (DS_OUT / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats))
    (DS_OUT / "source_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    # Compute exact fresh train-only norm_stats on 21D state (symmetric [-0.25,+0.25] for dims 14:21) + 16-step relative actions
    print("[PASS 3B] Computing fresh norm_stats_v3_fresh.json...", flush=True)
    st_all = np.concatenate(all_states, axis=0)
    ac_all = np.concatenate(all_rel_actions, axis=0)
    norm = {}
    for key, arr in (("state", st_all), ("actions", ac_all)):
        d = arr.shape[1]
        mean = arr.mean(axis=0, dtype=np.float64).tolist()
        std = arr.std(axis=0, dtype=np.float64).tolist()
        q01 = [float(np.quantile(arr[:, j], 0.01)) for j in range(d)]
        q99 = [float(np.quantile(arr[:, j], 0.99)) for j in range(d)]
        if key == "state":
            # Enforce symmetric [-0.25, +0.25] quantiles on home-delta dims 14:21 (matching EXP12)
            for j in range(14, 21):
                q01[j] = -0.25
                q99[j] = 0.25
        else:
            q01[7] = 0.0
            q99[7] = 1.0
        for j in range(d):
            assert np.isfinite(mean[j]) and np.isfinite(std[j]) and std[j] > 1e-8, f"Degenerate norm_stats {key}[{j}]"
            assert q99[j] > q01[j], f"Degenerate quantiles {key}[{j}]"
        norm[key] = dict(mean=mean, std=std, q01=q01, q99=q99)

    payload = json.dumps(dict(norm_stats=norm), indent=2) + "\n"
    NORM_OUT_NVME.parent.mkdir(parents=True, exist_ok=True)
    NORM_OUT_DURABLE.parent.mkdir(parents=True, exist_ok=True)
    NORM_OUT_NVME.write_text(payload)
    NORM_OUT_DURABLE.write_text(payload)
    sha = hashlib.sha256(payload.encode()).hexdigest()
    print(f"[NORM STATS READY] sha256={sha[:16]} -> {NORM_OUT_DURABLE}", flush=True)
    print(f"  actions std[:8] = {[round(x, 5) for x in norm['actions']['std']]}", flush=True)
    print(f"  actions q99[:8] = {[round(x, 5) for x in norm['actions']['q99']]}", flush=True)
    print(
        f"[DATASET READY] {DS_OUT}: episodes={ep_idx} (full={len(ok_full)}, surg={surg_count}), "
        f"frames={global_idx}, tasks={task_frame_counts}, elapsed={time.time()-t0:.1f}s",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--egl-gpus", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    RAW_FULL.mkdir(parents=True, exist_ok=True)
    RAW_SURG.mkdir(parents=True, exist_ok=True)
    train_scenes = json.loads(SCENES_MANIFEST.read_text())
    val_scenes = json.loads((H2 / "scenes_v3_real/stage2_validation_scenes_v3.json").read_text())
    test_scenes = json.loads((H2 / "scenes_v3_real/stage2_test_scenes_v3.json").read_text())
    forbidden = {int(s["seed"]) for s in val_scenes} | {int(s["seed"]) for s in test_scenes}
    for mf in (EXP13 / "validation_manifest.json", EXP13 / "test_manifest.json"):
        if mf.exists():
            forbidden |= {int(r["parent_scene_seed"]) for r in json.loads(mf.read_text())}
    train_seeds = {int(s["seed"]) for s in train_scenes}
    assert train_seeds.isdisjoint(forbidden), "Train/val/test overlap!"
    by_seed = {int(s["seed"]): s for s in train_scenes}

    if args.smoke:
        # Test 1 existing scene + 1 new superset scene!
        smoke_scs = [train_scenes[0], train_scenes[-1]]
        r0 = _collect_full_one((0, smoke_scs[0], args.egl_gpus))
        print("[SMOKE FULL orig]", r0, flush=True)
        assert r0["ok"], f"Smoke full orig failed: {r0}"
        r1 = _collect_full_one((1, smoke_scs[1], args.egl_gpus))
        print("[SMOKE FULL superset]", r1, flush=True)
        assert r1["ok"], f"Smoke full superset failed: {r1}"
        sd0 = int(r0["seed"])
        fams = [
            "unimodal_raise_approach", "approach_recovery",
            "p4_rec_tight", "p4_rec_core", "p4_rec_wide",
            "p7_rec_lip", "p7_rec_core", "p7_rec_wide",
            "unimodal_clear_home", "symmetric_home_brake_p11",
        ]
        for k, fam in enumerate(fams):
            job = dict(
                job_id=f"{fam}_000_s{sd0}", family=fam, idx=0,
                seed=sd0, rec_path=r0["path"], scene=by_seed[sd0],
            )
            rs = _collect_surgical_one((k, job, args.egl_gpus))
            print(f"[SMOKE SURG {fam:24s}]", rs, flush=True)
            assert rs["ok"], f"Smoke surgical {fam} failed: {rs}"
        print("[SMOKE ALL 11 FAMILIES (speedup=1.5, depth=+8mm, NVMe) PASSED 100%!]", flush=True)
        return

    print(f"[PASS 1] Collecting {len(train_scenes)} full cycles (speedup={MOTION_SPEEDUP}, depth=+8mm) -> {RAW_FULL} ({args.workers} workers, {args.egl_gpus} EGL GPUs)...", flush=True)
    ctx = mp.get_context("spawn")
    full_res = []
    ok_n = 0
    jobs1 = [(i, sc, args.egl_gpus) for i, sc in enumerate(train_scenes)]
    with ctx.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_collect_full_one, jobs1), 1):
            full_res.append(res)
            if res.get("ok"):
                ok_n += 1
            if i % 50 == 0 or i == len(train_scenes):
                print(f"  [PASS 1 {i}/{len(train_scenes)}] ok={ok_n} ({ok_n/i*100:.1f}%) elapsed={time.time()-t0:.1f}s", flush=True)
    (RAW_FULL / "manifest.json").write_text(json.dumps(full_res, indent=2) + "\n")
    ok_full = sorted([r for r in full_res if r.get("ok")], key=lambda r: int(r["seed"]))
    print(f"[PASS 1 DONE] {len(ok_full)}/{len(train_scenes)} ok in {time.time()-t0:.1f}s", flush=True)
    assert len(ok_full) >= 950, f"Too few Pass 1 episodes: {len(ok_full)}"

    specs = [
        ("unimodal_raise_approach", 250, 0),
        ("approach_recovery", 250, 3),
        ("p4_rec_tight", 225, 7),
        ("p4_rec_core", 495, 11),
        ("p4_rec_wide", 180, 17),
        ("p7_rec_lip", 240, 23),
        ("p7_rec_core", 270, 29),
        ("p7_rec_wide", 90, 31),
        ("unimodal_clear_home", 150, 37),
        ("symmetric_home_brake_p11", 150, 41),
    ]
    n_ok = len(ok_full)
    surg_jobs = []
    for fam, count, off in specs:
        idxs = np.linspace(0, n_ok - 1, count, dtype=int)
        for k, idx_v in enumerate(idxs):
            rec = ok_full[(int(idx_v) + off) % n_ok]
            sd = int(rec["seed"])
            surg_jobs.append((len(surg_jobs), dict(
                job_id=f"{fam}_{k:03d}_s{sd}", family=fam, idx=k,
                seed=sd, rec_path=rec["path"], scene=by_seed[sd],
            ), args.egl_gpus))
    print(f"[PASS 2] Collecting {len(surg_jobs)} surgical + 3-tier P4/P7 recovery episodes -> {RAW_SURG}...", flush=True)
    t2 = time.time()
    surg_res = []
    sok = 0
    with ctx.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_collect_surgical_one, surg_jobs), 1):
            surg_res.append(res)
            if res.get("ok"):
                sok += 1
            if i % 100 == 0 or i == len(surg_jobs):
                print(f"  [PASS 2 {i}/{len(surg_jobs)}] ok={sok} ({sok/i*100:.1f}%) elapsed={time.time()-t2:.1f}s", flush=True)
    (RAW_SURG / "manifest.json").write_text(json.dumps(surg_res, indent=2) + "\n")
    print(f"[PASS 2 DONE] {sok}/{len(surg_jobs)} ok in {time.time()-t2:.1f}s", flush=True)

    print("[PASS 3] Building LeRobot v2.1 train_v3_fresh + fresh norm_stats_v3_fresh.json on local NVMe...", flush=True)
    build_lerobot_and_norm_stats(ok_full, forbidden)


if __name__ == "__main__":
    main()
