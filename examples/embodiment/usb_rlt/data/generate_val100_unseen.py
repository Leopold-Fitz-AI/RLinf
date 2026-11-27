# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Generate 100 completely unseen benchmark-distribution validation scenes (seeds 700001..700100)."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

for p in [
    "/root/v12b_usb_cycle_v1/runtime",
    "/vla-cd/jy/v12b_usb_cycle_v1/holder_v2_rlt/code",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import generate_v3_real_train_scenes as gen


def main() -> None:
    h2 = Path("/vla-cd/jy/v12b_usb_cycle_v1/holder_v2_rlt")
    out_dir = Path("/mlplatform/jy/v3_fresh_nvme/val100_unseen")
    out_dir.mkdir(parents=True, exist_ok=True)

    forbid: set[int] = set()
    for fn in [
        h2 / "scenes_v3_real/stage2_train_scenes_v3.json",
        h2 / "scenes_v3_real/stage2_validation_scenes_v3.json",
        h2 / "scenes_v3_real/stage2_test_scenes_v3.json",
        Path("/mlplatform/jy/v3_fresh_nvme/train_scenes_v3_1200.json"),
    ]:
        forbid |= {int(x["seed"]) for x in json.loads(fn.read_text())}

    seeds = list(range(700001, 700101))
    assert set(seeds).isdisjoint(forbid), "SEED OVERLAP DETECTED!"

    t0 = time.time()
    jobs = [(sd, str(out_dir), 0.010, 5.0) for sd in seeds]
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(processes=32) as pool:
        for res in pool.imap_unordered(gen._gen_one, jobs):
            assert res.get("ok"), f"Gen failed: {res}"
            rng_b = np.random.default_rng(int(res["seed"]) + 99991)
            br = float(rng_b.uniform(0.85, 1.15))
            rows.append(
                dict(
                    seed=int(res["seed"]),
                    scene=res["scene"],
                    brightness=br,
                    split="validation_unseen100",
                    source="v3_val100_unseen_10mm_5deg",
                )
            )
    rows.sort(key=lambda x: x["seed"])
    out_m = h2 / "scenes_v3_real/val100_unseen_scenes_v3.json"
    out_m.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"GENERATED val100_unseen: {len(rows)} scenes in {time.time()-t0:.1f}s -> {out_m}", flush=True)


if __name__ == "__main__":
    main()
