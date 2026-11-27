# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Train lightweight (<0.11M fp32) Stage 2 RLT Transit head (p56_rlt_v2) for:
  - P5 (pure axial +Z withdraw with zero lateral shear), and
  - P6 (high-precision return to robot's own P2 encoder memory q_p2_lift).

Strictly robot-only proprioception (q_cur, q_p2_lift_mem, tcp_cur, tcp_entry) +
frozen Stage 1B z_rl features. Zero privileged pose, zero val/test leakage.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/vla-cd/jy/v12b_usb_cycle_v1")
H2 = ROOT / "holder_v2_rlt"
EXP12 = ROOT / "clean_rel256_phase12_v1"
EXP13 = ROOT / "clean_rel256_phase13_v1"

for p in [
    "/root/v12b_usb_cycle_v1/runtime",
    str(H2 / "code"),
    str(EXP12 / "code"),
    str(EXP13 / "code"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)


class Stage2P56ResidualActor(nn.Module):
    """Unified 0.104M fp32 Stage 2 RLT actor for P5 (6-DoF axial withdraw) and P6 (7-DoF return to q_p2_lift).

    Inputs (all strictly non-privileged robot proprioception + frozen Stage 1B z_rl):
      - z_rl: (B, 2048) frozen RLTTokenTransformer visual-state embedding
      - feat: (B, 36) = [phase_onehot(2: P5/P6), q_cur(7), q_goal_mem(7),
                         (q_goal_mem - q_cur)(7), bounded_dir(7), tcp_cur(3), (tcp_cur - tcp_entry)(3)]
    Outputs:
      - act7: (B, 7) in [-1, 1]:
          * In P5 (phase=5): act7[:, :6] is 6-DoF Cartesian delta (fed to URDF Jacobian)
          * In P6 (phase=6): act7[:, :7] is 7-DoF joint delta (scaled by max_dq_p6 / 0.96)
    """

    def __init__(self, z_dim: int = 2048, feat_dim: int = 36, hidden: int = 128) -> None:
        super().__init__()
        self.z_proj = nn.Sequential(nn.Linear(z_dim, 32), nn.LayerNorm(32), nn.SiLU())
        self.f_proj = nn.Sequential(nn.Linear(feat_dim, 64), nn.LayerNorm(64), nn.SiLU())
        self.trunk = nn.Sequential(
            nn.Linear(96, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 7)

    def forward(self, z_rl: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.z_proj(z_rl), self.f_proj(feat)], dim=-1)
        return torch.tanh(self.head(self.trunk(h)))


def make_p56_feat(
    phase_idx: int,
    q_cur: np.ndarray,
    q_goal_mem: np.ndarray,
    tcp_cur: np.ndarray,
    tcp_entry: np.ndarray,
    *,
    max_dq_joint: float = 0.025,
) -> np.ndarray:
    """Build strictly robot-only 36D feature vector for Stage2P56ResidualActor (P5 / P6)."""
    oh = np.array([1.0, 0.0] if phase_idx == 5 else [0.0, 1.0], dtype=np.float32)
    qc = np.asarray(q_cur, dtype=np.float32).reshape(7)
    qg = np.asarray(q_goal_mem, dtype=np.float32).reshape(7)
    dq = np.clip(qg - qc, -2.5, 2.5)
    if phase_idx == 5:
        bdir = np.zeros(7, dtype=np.float32)
    else:
        max_abs = max(float(np.max(np.abs(dq))), 1e-6)
        scale = min(1.0, float(max_dq_joint) / max_abs)
        bdir = np.clip((dq * scale) / max(float(max_dq_joint), 1e-6), -1.0, 1.0).astype(np.float32)
    tc = np.asarray(tcp_cur, dtype=np.float32).reshape(3)
    dt = np.clip(tc - np.asarray(tcp_entry, dtype=np.float32).reshape(3), -0.5, 0.5)
    return np.concatenate([oh, qc, qg, dq, bdir, tc, dt], axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4-buf", default="/mlplatform/jy/v3_fresh_nvme/stage2_buffers/p4_v3_tristereo640_demo_buf.pt")
    ap.add_argument("--p7-buf", default="/mlplatform/jy/v3_fresh_nvme/stage2_buffers/p7_v3_real_tristereo640_demo_buf.pt")
    ap.add_argument("--train-pool-p7", default=str(H2 / "handoff_pools_v3_fresh/p7_real_pool_train.json"))
    ap.add_argument("--out-pt", default=str(H2 / "stage2_rlt_opt3a/frozen_p56_v2_rlt_head.pt"))
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--max-dq-p6", type=float, default=0.025)
    ap.add_argument("--p5-dz-act", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")

    b4 = torch.load(args.p4_buf, map_location="cpu", weights_only=False)
    z4 = np.asarray(b4["z_table"], dtype=np.float32)
    del b4

    b7 = torch.load(args.p7_buf, map_location="cpu", weights_only=False)
    z7 = np.asarray(b7["z_table"], dtype=np.float32)
    del b7

    p7_tr = json.loads(Path(args.train_pool_p7).read_text())
    q_dock_s, q_lift_s, q_home_s = [], [], []
    tcp_dock_s, tcp_lift_s, tcp_home_s = [], [], []
    for it in p7_tr:
        seed = int(it["scene"]["seed"])
        demo = Path("/mlplatform/jy/v3_fresh_nvme/raw_full_1200") / str(seed) / "demonstration_full.npz"
        if not demo.exists():
            demo = H2 / "raw_v3_real" / str(seed) / "demonstration_full.npz"
        if not demo.exists():
            continue
        with np.load(demo, allow_pickle=True) as d:
            st = d["observation.state"]
            dock_mask = (st[:, 1] > -0.18) & (st[:, 2] < 0.848)
            lift_mask = (st[:, 1] < -0.24) & (st[:, 2] > 0.820) & (st[:, 2] < 0.845)
            p0_idx = np.linspace(0, min(260, len(st) - 1), 16, dtype=int)
            q_home_s.append(st[p0_idx, 6:13].astype(np.float32))
            tcp_home_s.append(st[p0_idx, :3].astype(np.float32))
            if np.any(dock_mask):
                idxs = np.where(dock_mask)[0]
                pick = idxs[rng.integers(0, len(idxs), size=min(8, len(idxs)))]
                q_dock_s.append(st[pick, 6:13].astype(np.float32))
                tcp_dock_s.append(st[pick, :3].astype(np.float32))
            if np.any(lift_mask):
                idxs = np.where(lift_mask)[0]
                pick = idxs[rng.integers(0, len(idxs), size=min(8, len(idxs)))]
                q_lift_s.append(st[pick, 6:13].astype(np.float32))
                tcp_lift_s.append(st[pick, :3].astype(np.float32))

    q_dock = np.concatenate(q_dock_s, axis=0)
    q_lift = np.concatenate(q_lift_s, axis=0)
    q_home = np.concatenate(q_home_s, axis=0)
    tcp_dock = np.concatenate(tcp_dock_s, axis=0)
    tcp_lift = np.concatenate(tcp_lift_s, axis=0)
    tcp_home = np.concatenate(tcp_home_s, axis=0)
    q_any = np.concatenate([q_dock, q_lift, q_home], axis=0)
    tcp_any = np.concatenate([tcp_dock, tcp_lift, tcp_home], axis=0)
    print(f"[DATA] z4={z4.shape} z7={z7.shape} q_dock={q_dock.shape} q_lift={q_lift.shape} q_any={q_any.shape}", flush=True)

    actor = Stage2P56ResidualActor().to(device)
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"[MODEL] Stage2P56ResidualActor (p56_rlt_v2) params = {n_params:,} ({n_params/1e6:.3f}M)", flush=True)
    opt = torch.optim.AdamW(actor.parameters(), lr=1.5e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)

    z4_t = torch.from_numpy(z4).to(device)
    z7_t = torch.from_numpy(z7).to(device)
    qd_t = torch.from_numpy(q_dock).to(device)
    ql_t = torch.from_numpy(q_lift).to(device)
    qa_t = torch.from_numpy(q_any).to(device)
    td_t = torch.from_numpy(tcp_dock).to(device)
    tl_t = torch.from_numpy(tcp_lift).to(device)
    ta_t = torch.from_numpy(tcp_any).to(device)

    B5 = args.batch_size // 2
    B6 = args.batch_size - B5
    max_dq6 = float(args.max_dq_p6)
    p5_dz = float(args.p5_dz_act)

    for step in range(1, args.steps + 1):
        # P5: pure axial +Z lift braking smoothly to 0 at +42mm above P5 entry
        iz5 = torch.randint(0, len(z4_t), (B5,), device=device)
        id5 = torch.randint(0, len(qd_t), (B5,), device=device)
        il5 = torch.randint(0, len(ql_t), (B5,), device=device)
        z5 = z4_t[iz5] + 0.05 * torch.randn((B5, 2048), device=device)
        qc5 = qd_t[id5] + 0.18 * torch.randn((B5, 7), device=device)
        ql5 = ql_t[il5] + 0.05 * torch.randn((B5, 7), device=device)
        dq5 = torch.clamp(ql5 - qc5, -2.5, 2.5)
        bdir5 = torch.zeros((B5, 7), device=device)
        tc5 = td_t[id5] + 0.02 * torch.randn((B5, 3), device=device)
        dt5 = torch.empty((B5, 3), device=device)
        dt5[:, :2] = 0.004 * torch.randn((B5, 2), device=device)
        dt5[:, 2] = torch.rand((B5,), device=device) * 0.055
        oh5 = torch.tensor([1.0, 0.0], device=device).expand(B5, 2)
        feat5 = torch.cat([oh5, qc5, ql5, dq5, bdir5, tc5, dt5], dim=-1)
        tgt5 = torch.zeros((B5, 7), device=device)
        tgt5[:, 2] = p5_dz * torch.clamp((0.042 - dt5[:, 2]) / 0.006, 0.0, 1.0)

        # P6 & P10: high-precision joint-space waypoint tracking across dock->q_p2_lift AND reversed P0->q_home0
        iz6 = torch.randint(0, len(z7_t), (B6,), device=device)
        id6 = torch.randint(0, len(qa_t), (B6,), device=device)
        il6 = torch.randint(0, len(qa_t), (B6,), device=device)
        z6 = z7_t[iz6] + 0.05 * torch.randn((B6, 2048), device=device)
        qg6 = qa_t[il6] + 0.08 * torch.randn((B6, 7), device=device)
        qs6 = qa_t[id6] + 0.25 * torch.randn((B6, 7), device=device)
        alpha = torch.rand((B6, 1), device=device)
        near = (torch.rand((B6, 1), device=device) < 0.40)
        alpha = torch.where(near, 0.90 + 0.10 * alpha, alpha)
        qc6 = (1.0 - alpha) * qs6 + alpha * qg6 + torch.where(near, 0.003, 0.015) * torch.randn((B6, 7), device=device)
        dq6 = torch.clamp(qg6 - qc6, -2.5, 2.5)
        max_abs = torch.clamp(dq6.abs().max(dim=-1, keepdim=True).values, min=1e-6)
        scale = torch.clamp(max_dq6 / max_abs, max=1.0)
        bdir6 = torch.clamp((dq6 * scale) / max_dq6, -1.0, 1.0)
        tgt6 = torch.clamp(bdir6 * 0.96, -0.96, 0.96)
        tc6 = (1.0 - alpha) * ta_t[id6] + alpha * ta_t[il6] + 0.02 * torch.randn((B6, 3), device=device)
        dt6 = torch.clamp(tc6 - ta_t[id6], -0.5, 0.5)
        oh6 = torch.tensor([0.0, 1.0], device=device).expand(B6, 2)
        feat6 = torch.cat([oh6, qc6, qg6, dq6, bdir6, tc6, dt6], dim=-1)

        z_all = torch.cat([z5, z6], dim=0)
        f_all = torch.cat([feat5, feat6], dim=0)
        y_all = torch.cat([tgt5, tgt6], dim=0)

        pred = actor(z_all, f_all)
        loss = F.smooth_l1_loss(pred, y_all, beta=0.004)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 2000 == 0 or step == args.steps:
            with torch.no_grad():
                mae5 = (pred[:B5, :6] - tgt5[:, :6]).abs().mean().item()
                mae6 = (pred[B5:] - tgt6).abs().mean().item()
            print(f"[STEP {step}/{args.steps}] loss={loss.item():.6f} mae5={mae5:.5f} mae6={mae6:.5f} elapsed={time.time()-t0:.1f}s", flush=True)

    out_path = Path(args.out_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            arch_version="p56_rlt_v2",
            actor=actor.state_dict(),
            n_params=n_params,
            max_dq_p6=max_dq6,
            p5_scale_xyz_m=0.0018,
            p5_scale_rot_rad=0.022,
            p5_q_clamp_rad=0.020,
            p5_dz_act=p5_dz,
            args=vars(args),
        ),
        out_path,
    )
    print(f"[DONE] Saved {out_path} ({n_params:,} params) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
