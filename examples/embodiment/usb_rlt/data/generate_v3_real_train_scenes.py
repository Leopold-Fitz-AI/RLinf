# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Generate 828 new high-density superset v3_real train scenes (+ 372 existing = 1,200 total).

Superset randomization on real_aligned_v3 (preserving existing 372 train + frozen 34 val / 68 test):
  - usb_holder & socket XY: [-13, +13] mm, yaw: [-6, +6] deg (1.3x benchmark superset)
  - brightness: [0.75, 1.25], friction: [0.80, 1.20], mass: [0.85, 1.15], home_q jitter: [-0.025, +0.025] rad
  - Strictly disjoint from validation (34) and test (68) seeds; verifies MuJoCo load & zero plug penetration.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

for p in [
    "/root/v12b_usb_cycle_v1/runtime",
    "/vla-cd/jy/v12b_usb_cycle_v1/holder_v2_rlt/code",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/vla-cd/jy/v12b_usb_cycle_v1")
H2 = ROOT / "holder_v2_rlt"
EXP13 = ROOT / "clean_rel256_phase13_v1"
CANONICAL_V3_XML = H2 / "scenes_v3_real/training_scene_v3.xml"
OUT_ROOT = Path("/mlplatform/jy/v3_fresh_nvme")


def _numbers(a) -> str:
    return " ".join(format(float(x), ".17g") for x in a)


def _rotate_quat(q, rot: Rotation) -> np.ndarray:
    return (rot * Rotation.from_quat(np.asarray(q, dtype=np.float64)[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]


def _gen_one(args: tuple[int, str, float, float]) -> dict:
    seed, out_dir_str, xy_range_m, yaw_range_deg = args
    out_dir = Path(out_dir_str)
    dst_xml = out_dir / str(seed) / "scene_v3.xml"
    meta_p = out_dir / str(seed) / "layout_v3.json"
    if dst_xml.exists() and meta_p.exists():
        m = json.loads(meta_p.read_text())
        if m.get("ok"):
            return m

    rng = np.random.default_rng(seed)
    tree = ET.parse(CANONICAL_V3_XML)
    root = tree.getroot()
    src_dir = CANONICAL_V3_XML.parent.resolve()

    # Ensure all asset file paths are absolute
    asset_el = root.find("asset")
    if asset_el is not None:
        for elem in asset_el:
            f = elem.get("file")
            if f and not Path(f).is_absolute():
                elem.set("file", str((src_dir / f).resolve()))

    changes = {}
    holder = root.find("worldbody/body[@name='usb_holder']")
    origin = np.fromstring(holder.get("pos"), sep=" ")
    for name in ("usb_holder", "socket"):
        body = root.find(f"worldbody/body[@name='{name}']")
        delta = np.r_[rng.uniform(-xy_range_m, xy_range_m, 2), 0.0]
        yaw = float(rng.uniform(-yaw_range_deg, yaw_range_deg))
        rot = Rotation.from_euler("z", yaw, degrees=True)
        body.set("pos", _numbers(np.fromstring(body.get("pos"), sep=" ") + delta))
        body.set("quat", _numbers(_rotate_quat(np.fromstring(body.get("quat", "1 0 0 0"), sep=" "), rot)))
        changes[name] = dict(translation_m=delta.tolist(), yaw_deg=yaw)
        if name == "usb_holder":
            plug = root.find("worldbody/body[@name='plug']")
            plug.set("pos", _numbers(origin + delta + rot.apply(np.fromstring(plug.get("pos"), sep=" ") - origin)))
            plug.set("quat", _numbers(_rotate_quat(np.fromstring(plug.get("quat"), sep=" "), rot)))
            key = root.find("keyframe/key[@name='home']")
            q = np.fromstring(key.get("qpos"), sep=" ")
            q[-7:-4] = origin + delta + rot.apply(q[-7:-4] - origin)
            q[-4:] = _rotate_quat(q[-4:], rot)
            key.set("qpos", _numbers(q))

    # Lookup friction values before scaling
    lookup = ET.fromstring(ET.tostring(root))
    lookup_geoms = list(lookup.find("worldbody").iter("geom"))
    for i, g in enumerate(lookup_geoms):
        if not g.get("name"):
            g.set("name", f"friction_lookup_{i}")
    m = mujoco.MjModel.from_xml_string(ET.tostring(lookup, encoding="unicode"))

    key = root.find("keyframe/key[@name='home']")
    q = np.fromstring(key.get("qpos"), sep=" ")
    assert m.joint("usb_free").qposadr[0] == m.nq - 7, "usb_free must be last 7 qpos"
    jnames = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ids = [m.joint("right_" + nm + "_joint").id for nm in jnames]
    aq = m.jnt_qposadr[ids]
    jitter = rng.uniform(-0.025, 0.025, 7)
    q[aq] = np.clip(q[aq] + jitter, m.jnt_range[ids, 0] + 0.001, m.jnt_range[ids, 1] - 0.001)
    key.set("ctrl", _numbers(q[m.jnt_qposadr[m.actuator_trnid[:, 0]]]))

    mass = float(rng.uniform(0.85, 1.15))
    friction = float(rng.uniform(0.80, 1.20))
    brightness = float(rng.uniform(0.75, 1.25))
    plug = root.find("worldbody/body[@name='plug']")
    iner = plug.find("inertial")
    iner.set("mass", str(float(iner.get("mass")) * mass))
    iner.set("diaginertia", _numbers(np.fromstring(iner.get("diaginertia"), sep=" ") * mass))

    for geom, lgeom in zip(root.find("worldbody").iter("geom"), lookup_geoms):
        gid = m.geom(lgeom.get("name")).id
        if m.geom_contype[gid] or m.geom_conaffinity[gid]:
            geom.set("friction", _numbers(m.geom_friction[gid] * friction))

    opt = root.find("option")
    opt.set("cone", "elliptic")
    opt.set("impratio", "10")

    # Collision-free plug reset proposal inside 4.80x12.40 mm holder slot
    probe = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(probe)
    body_id = probe.body("plug").id
    accepted = None
    for attempt in range(64):
        cand = q.copy()
        d_plug = np.r_[rng.uniform(-0.00010, 0.00010, 2), rng.uniform(-0.0012, 0.0012)]
        y_plug = float(rng.uniform(-1.2, 1.2))
        cand[-7:-4] += d_plug
        cand[-4:] = _rotate_quat(q[-4:], Rotation.from_euler("z", y_plug, degrees=True))
        mujoco.mj_resetDataKeyframe(probe, d, probe.key("home").id)
        d.qpos[:] = cand
        mujoco.mj_forward(probe, d)
        ov = [float(c.dist) for c in d.contact if body_id in probe.geom_bodyid[c.geom] and c.dist < -1e-5]
        if not ov:
            accepted = dict(attempt=attempt, delta_xyz_m=d_plug.tolist(), yaw_deg=y_plug)
            break
    if accepted is None:
        cand = q.copy()
        mujoco.mj_resetDataKeyframe(probe, d, probe.key("home").id)
        d.qpos[:] = cand
        mujoco.mj_forward(probe, d)
        ov = [float(c.dist) for c in d.contact if body_id in probe.geom_bodyid[c.geom] and c.dist < -1e-5]
        if ov:
            return dict(seed=seed, ok=False, err=f"nominal_overlap={min(ov):.6f}")
        accepted = dict(attempt=64, delta_xyz_m=[0.0, 0.0, 0.0], yaw_deg=0.0, nominal_fallback=True)

    key.set("qpos", _numbers(cand))
    plug.set("pos", _numbers(cand[-7:-4]))
    plug.set("quat", _numbers(cand[-4:]))

    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(str(dst_xml), encoding="utf-8", xml_declaration=False)

    # Final load sanity check
    _ = mujoco.MjModel.from_xml_path(str(dst_xml))
    rec = dict(
        seed=seed,
        ok=True,
        scene=str(dst_xml),
        brightness=brightness,
        friction=friction,
        mass=mass,
        split="train",
        source="v3_superset_13mm_6deg",
        changes=changes,
        reset_proposal=accepted,
    )
    meta_p.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-train", type=int, default=1200)
    ap.add_argument("--xy-range-mm", type=float, default=13.0)
    ap.add_argument("--yaw-range-deg", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    existing_train = json.loads((H2 / "scenes_v3_real/stage2_train_scenes_v3.json").read_text())
    val_scenes = json.loads((H2 / "scenes_v3_real/stage2_validation_scenes_v3.json").read_text())
    test_scenes = json.loads((H2 / "scenes_v3_real/stage2_test_scenes_v3.json").read_text())

    forbidden = {int(s["seed"]) for s in val_scenes} | {int(s["seed"]) for s in test_scenes}
    for mf in (EXP13 / "validation_manifest.json", EXP13 / "test_manifest.json"):
        if mf.exists():
            forbidden |= {int(r["parent_scene_seed"]) for r in json.loads(mf.read_text())}
    existing_seeds = {int(s["seed"]) for s in existing_train}
    assert existing_seeds.isdisjoint(forbidden), "Existing train overlaps val/test!"

    need = max(0, args.total_train - len(existing_train))
    new_seeds = []
    cand = 613000
    while len(new_seeds) < need:
        if cand not in forbidden and cand not in existing_seeds:
            new_seeds.append(cand)
        cand += 1
        assert cand <= 619999, "Exhausted train seed range!"

    out_dir = OUT_ROOT / "scenes_train_1200"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(new_seeds)} new v3_real superset scenes (±{args.xy_range_mm}mm, ±{args.yaw_range_deg}deg) -> {out_dir}...", flush=True)
    jobs = [(sd, str(out_dir), args.xy_range_mm / 1000.0, args.yaw_range_deg) for sd in new_seeds]
    ctx = mp.get_context("spawn")
    new_rows = []
    with ctx.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_gen_one, jobs), 1):
            assert res.get("ok"), f"Scene generation failed: {res}"
            new_rows.append(res)
            if i % 200 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] generated ok in {time.time()-t0:.1f}s", flush=True)

    combined = [dict(s, source="v3_orig_372") for s in existing_train] + [
        dict(seed=int(r["seed"]), scene=r["scene"], brightness=float(r["brightness"]), split="train", source=r["source"])
        for r in sorted(new_rows, key=lambda x: int(x["seed"]))
    ]
    all_seeds = {int(s["seed"]) for s in combined}
    assert len(all_seeds) == len(combined) == args.total_train, f"Count mismatch: {len(all_seeds)} vs {args.total_train}"
    assert all_seeds.isdisjoint(forbidden), "LEAKAGE detected in generated scenes!"

    out_manifest = OUT_ROOT / "train_scenes_v3_1200.json"
    out_manifest.write_text(json.dumps(combined, indent=2) + "\n")
    # Also persist a copy on /vla-cd for durability
    durable_manifest = H2 / "scenes_v3_real/train_scenes_v3_1200_manifest.json"
    durable_manifest.write_text(json.dumps(combined, indent=2) + "\n")
    print(f"[SCENES READY] total={len(combined)} (372 orig + {len(new_rows)} superset), zero val/test overlap, elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
