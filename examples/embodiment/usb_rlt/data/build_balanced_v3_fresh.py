# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Build zero-copy phase-balanced LeRobot v2.1 dataset /mlplatform/jy/v3_fresh_nvme/train_v3_fresh_bal.

Fixes P0/P10/P11 under-representation (3.5%/1.8%/1.0% -> ~9%/5%/4%) by symlinking existing
parquets & videos from train_v3_fresh with balanced repeats, keeping the EXACT same
assets/v3_fresh/norm_stats.json locked!
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.common.datasets.utils import create_empty_dataset_info

import sys
for p in [
    "/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase12_v1/code",
    "/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/code",
    "/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/dependencies",
    "/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/rlinf",
]:
    if p not in sys.path:
        sys.path.insert(0, p)
import usb_rel256_phase12 as usb12

NVME = Path("/mlplatform/jy/v3_fresh_nvme")
SRC_DS = NVME / "train_v3_fresh"
DST_DS = NVME / "train_v3_fresh_bal"
CAMERAS = ("image", "left_wrist_image", "wrist_image")

REPEAT_BY_KEY = {
    "v3_fresh_p1_to_p9": 1,
    "v3_fresh_p4_anchor": 1,
    "v3_fresh_p7_anchor": 0,       # drop redundant P7 anchor (already 600 P7 3-tier recovery + 1,049 full cycles!)
    "unimodal_raise_approach": 3,  # boost P0 from 3.5% -> ~9%
    "approach_recovery": 2,        # boost P1 visual centering
    "p4_rec_tight": 1,
    "p4_rec_core": 1,
    "p4_rec_wide": 1,
    "p7_rec_lip": 1,
    "p7_rec_core": 1,
    "p7_rec_wide": 1,
    "unimodal_clear_home": 3,      # boost P9/P10/P11
    "symmetric_home_brake_p11": 4, # boost P11 braking
}


def main() -> None:
    t0 = time.time()
    shutil.rmtree(DST_DS, ignore_errors=True)
    (DST_DS / "meta").mkdir(parents=True, exist_ok=True)

    src_info = json.loads((SRC_DS / "meta/info.json").read_text())
    features = src_info["features"]
    src_eps = [json.loads(l) for l in (SRC_DS / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    src_stats = [json.loads(l) for l in (SRC_DS / "meta/episodes_stats.jsonl").read_text().splitlines() if l.strip()]
    src_prov = json.loads((SRC_DS / "source_provenance.json").read_text())
    assert len(src_eps) == len(src_stats) == len(src_prov)

    chunk_size = 1000
    episodes, stats, provenance = [], [], []
    task_counts = [0] * len(usb12.PHASE_PROMPTS)
    ep_idx, global_idx = 0, 0

    for old_i, (ep_m, st_m, pr_m) in enumerate(zip(src_eps, src_stats, src_prov)):
        key = pr_m.get("family") or pr_m.get("source")
        rep = int(REPEAT_BY_KEY.get(key, 1))
        if rep <= 0:
            continue
        old_ch = old_i // chunk_size
        src_pq = SRC_DS / f"data/chunk-{old_ch:03d}/episode_{old_i:06d}.parquet"
        tbl = pq.read_table(src_pq)
        n = len(tbl)
        ti_col = tbl["task_index"].to_pylist()
        ep_col_i = tbl.schema.get_field_index("episode_index")
        idx_col_i = tbl.schema.get_field_index("index")
        vid_srcs = {c: (SRC_DS / f"videos/chunk-{old_ch:03d}/{c}/episode_{old_i:06d}.mp4").resolve() for c in CAMERAS}

        for r_k in range(rep):
            ch = ep_idx // chunk_size
            (DST_DS / f"data/chunk-{ch:03d}").mkdir(parents=True, exist_ok=True)
            for c, vs in vid_srcs.items():
                dst_v = DST_DS / f"videos/chunk-{ch:03d}/{c}/episode_{ep_idx:06d}.mp4"
                dst_v.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(vs), str(dst_v))
            tbl_out = tbl.set_column(
                ep_col_i, tbl.schema.field(ep_col_i), pa.array(np.full(n, ep_idx, dtype=np.int64))
            ).set_column(
                idx_col_i, tbl.schema.field(idx_col_i), pa.array(np.arange(global_idx, global_idx + n, dtype=np.int64))
            )
            pq.write_table(tbl_out, str(DST_DS / f"data/chunk-{ch:03d}/episode_{ep_idx:06d}.parquet"))
            for t_v in ti_col:
                task_counts[int(t_v)] += 1
            episodes.append(dict(episode_index=ep_idx, tasks=ep_m["tasks"], length=n))
            new_st = dict(st_m["stats"])
            new_st["episode_index"] = dict(min=[float(ep_idx)], max=[float(ep_idx)], mean=[float(ep_idx)], std=[0.0], count=[n])
            idx_a = np.arange(global_idx, global_idx + n, dtype=np.float64)
            new_st["index"] = dict(min=[float(idx_a[0])], max=[float(idx_a[-1])], mean=[float(idx_a.mean())], std=[float(idx_a.std())], count=[n])
            stats.append(dict(episode_index=ep_idx, stats=new_st))
            provenance.append(dict(**{k: v for k, v in pr_m.items() if k != "episode_index"}, episode_index=ep_idx, repeat=r_k))
            global_idx += n
            ep_idx += 1

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
    (DST_DS / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (DST_DS / "meta/tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": p}) + "\n" for i, p in enumerate(usb12.PHASE_PROMPTS))
    )
    (DST_DS / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (DST_DS / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats))
    (DST_DS / "source_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    pcts = [round(c / global_idx * 100, 1) for c in task_counts]
    print(f"[BALANCED READY] {DST_DS}: eps={ep_idx}, frames={global_idx}, pcts={pcts}, elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
