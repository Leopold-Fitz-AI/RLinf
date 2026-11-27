# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Export an RLinf FSDP dcp_checkpoint to a standalone full_weights.pt warmstart directory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

for p in [
    "/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase12_v1/code",
    "/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/code",
    "/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/dependencies",
    "/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1/rlinf",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.distributed.checkpoint as dcp
from hydra import compose, initialize_config_dir
from rlinf.models.embodiment.openpi import get_model
import usb_rel256_phase12 as usb12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--dcp-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    usb12.register()
    with initialize_config_dir(config_dir=str(Path(args.config_dir).resolve()), version_base="1.1"):
        cfg = compose(config_name=args.config_name)
    model = get_model(cfg.actor.model, torch_dtype=torch.float32).eval()
    state = {"fsdp_checkpoint": {"model": model.state_dict()}}
    dcp.load(state, checkpoint_id=str(args.dcp_dir))
    sd = state["fsdp_checkpoint"]["model"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / "full_weights.pt"
    tmp_pt = out_dir / "full_weights.pt.tmp"
    torch.save(sd, str(tmp_pt))
    tmp_pt.replace(out_pt)
    print(f"[EXPORTED] {len(sd)} tensors -> {out_pt} ({out_pt.stat().st_size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
