# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run any H2/configs/<config_name>.yaml in an owned isolated Ray cluster (8 GPUs if 4-7 free, else 4 GPUs)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

EXP12 = Path("/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase12_v1")
EXP13 = Path("/vla-cd/jy/v12b_usb_cycle_v1/clean_rel256_phase13_v1")
H2 = Path("/vla-cd/jy/v12b_usb_cycle_v1/holder_v2_rlt")
PYTHON = "/root/g1_usb_sim2real/.venv-pi05/bin/python"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def gpus_4_to_7_idle() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        used = {}
        for line in out.strip().splitlines():
            idx_s, mem_s = [x.strip() for x in line.split(",")]
            used[int(idx_s)] = int(mem_s)
        return all(used.get(g, 99999) < 2000 for g in (4, 5, 6, 7))
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="pi05_v3_fresh_sft")
    ap.add_argument("--gpus", default="", help="Explicit comma-separated GPU IDs, e.g. 4,5,6,7")
    ap.add_argument("--micro-batch-size", type=int, default=0, help="Override actor.micro_batch_size if > 0")
    args = ap.parse_args()

    name = args.config_name
    if args.gpus:
        gpu_list = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
        num_gpus = len(gpu_list)
        # Keep physical GPU indices visible to our isolated Ray cluster so RLinf component_placement
        # places workers onto the exact requested physical GPUs (e.g. 4-7)!
        ray_gpu_str = "0,1,2,3,4,5,6,7"
        ray_num_gpus = 8
        if gpu_list == list(range(gpu_list[0], gpu_list[-1] + 1)):
            placement_str = f"{gpu_list[0]}-{gpu_list[-1]}"
        else:
            placement_str = ",".join(str(g) for g in gpu_list)
        gpu_str = ",".join(str(g) for g in gpu_list)
        use_8 = (num_gpus == 8)
    else:
        use_8 = gpus_4_to_7_idle()
        num_gpus = 8 if use_8 else 4
        ray_num_gpus = num_gpus
        gpu_str = "0,1,2,3,4,5,6,7" if use_8 else "0,1,2,3"
        ray_gpu_str = gpu_str
        placement_str = "0-7" if use_8 else "0-3"
    port = free_port()
    ray_tmp = f"/dev/shm/jy_h2_ray_{uuid.uuid4().hex[:8]}"
    for d in [ray_tmp, "/dev/shm/jy_h2_tmp", "/dev/shm/jy_h2_triton", "/dev/shm/jy_h2_inductor"]:
        os.makedirs(d, exist_ok=True)
    (H2 / "logs").mkdir(parents=True, exist_ok=True)
    py_path = ":".join([str(EXP12 / "code"), str(EXP13 / "code"), str(EXP13 / "dependencies"), str(EXP13 / "rlinf")])
    env = dict(
        os.environ,
        RLINF_EXT_MODULE="usb_rel256_phase12",
        PYTHONPATH=py_path,
        CUDA_VISIBLE_DEVICES=ray_gpu_str,
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="4",
        RAY_TMPDIR=ray_tmp,
        TMPDIR="/dev/shm/jy_h2_tmp",
        TRITON_CACHE_DIR="/dev/shm/jy_h2_triton",
        TORCHINDUCTOR_CACHE_DIR="/dev/shm/jy_h2_inductor",
        TORCH_COMPILE_DISABLE="1",
        HF_HOME=str(H2 / ".cache/hf"),
        TORCH_HOME=str(H2 / ".cache/torch"),
        XDG_CACHE_HOME=str(H2 / ".cache/xdg"),
    )
    ray_log = open(H2 / f"logs/{name}_ray.log", "w")
    ray_proc = subprocess.Popen(
        [
            PYTHON, "-m", "ray.scripts.scripts", "start", "--head",
            "--node-ip-address=127.0.0.1", f"--port={port}", f"--num-gpus={ray_num_gpus}",
            "--include-dashboard=false", "--disable-usage-stats", f"--temp-dir={ray_tmp}", "--block",
        ],
        env=env, stdout=ray_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    time.sleep(8)
    env["RAY_ADDRESS"] = f"127.0.0.1:{port}"
    cmd = [
        PYTHON, str(EXP13 / "rlinf/examples/sft/train_vla_sft.py"),
        "--config-path", str(H2 / "configs"), "--config-name", name,
    ]
    if not use_8:
        mbs = args.micro_batch_size if args.micro_batch_size > 0 else (8 if "rlt" in name else 16)
        cmd.extend([f"cluster.component_placement.actor={placement_str}", f"actor.micro_batch_size={mbs}"])
    elif args.micro_batch_size > 0:
        cmd.append(f"actor.micro_batch_size={args.micro_batch_size}")
    t0 = time.time()
    status_path = H2 / f"stage1_pi05_rlt/{name}_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(dict(state="running", num_gpus=num_gpus, gpus=gpu_str, started=t0, ray_address=env["RAY_ADDRESS"]), indent=2) + "\n"
    )
    print(f"[LAUNCH] {name} on {num_gpus} GPUs ({gpu_str}), ray={env['RAY_ADDRESS']}", flush=True)
    try:
        with open(H2 / f"logs/{name}.log", "w") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            rc = proc.wait()
        status_path.write_text(
            json.dumps(dict(state=("completed" if rc == 0 else "failed"), num_gpus=num_gpus, returncode=rc, started=t0, ended=time.time()), indent=2) + "\n"
        )
        if rc != 0:
            raise SystemExit(rc)
    finally:
        try:
            os.killpg(os.getpgid(ray_proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(2)
        try:
            os.killpg(os.getpgid(ray_proc.pid), signal.SIGKILL)
        except Exception:
            pass
        shutil.rmtree(ray_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
