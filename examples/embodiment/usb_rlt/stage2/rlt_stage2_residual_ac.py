# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""RLT Stage 2: Stereo SpatialSoftmax + Contact-Proprio Persistent Compliant q_tgt RLPD.

Follows official RLinf RLT Stage 2 (frozen Stage 1 pi0.5 + RLTTokenTransformer, Twin-Q TD3/RLPD,
BC-human/HG-DAgger intervention loss, AWR, training-only stalled_progress expert takeover strictly
OFF at eval/test) + stereo wrist+front high-res SpatialSoftmax + training-only auxiliary alignment
supervision & asymmetric Twin-Q critic (NEVER read at inference/eval/test).

Strictly ZERO privileged pose at inference/eval/test: Actor consumes ONLY standard ACTOR_KEYS
(3x224x224 RGB -> z_rl + centered wrist 128x128 crop + centered front 96x96 crop, 21D proprio ->
ref_rel32) + real G1/Dex1-1 contact proprio (44D: fq_margin, q_err, tau_act, dq_recent, delta_tcp,
stall_steps, prev_act6).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import shutil
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from openpi import transforms as T
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
from torch.distributed import checkpoint as dcp
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
import robot_contact_utils as rcu
from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenTransformer
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.modules import model as openpi_model
from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms
import usb_rel256_phase12 as usb12


def crop_wrist_128(wrist_rgb_224: np.ndarray) -> np.ndarray:
    """Centered 128x128 crop around USB tip & socket/slot (u=112, v=160) -> uint8 CHW (3, 128, 128)."""
    img = np.asarray(wrist_rgb_224, dtype=np.uint8)
    crop = img[96:224, 48:176, :]  # v in [96, 224], u in [48, 176]
    return np.ascontiguousarray(crop.transpose(2, 0, 1))


def render_wrist_hires_128(task) -> np.ndarray:
    """Option 3A native MuJoCo 640x640 -> 128x128 centered wrist ROI (~2.25 px/mm, zero privilege).

    Matches RealSense D405 native optical resolution and ObserverObservation camera/brightness contract:
    tip=(320, 461), P4 port=(320, 452), P7 holder=(319, 437) all sit dead-center inside [376:504, 256:384].
    """
    rend = getattr(task, "_hires_wrist_renderer", None)
    if rend is None:
        rend = mujoco.Renderer(task.m, 640, 640)
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        task._hires_wrist_renderer = rend
        task._hires_wrist_opt = opt
    rend.update_scene(task.d, camera="wrist", scene_option=task._hires_wrist_opt)
    raw = rend.render()
    crop = raw[376:504, 256:384, :]
    br = float(getattr(task, "brightness", 1.0))
    if br != 1.0:
        crop = np.clip(crop.astype(np.float32) * br, 0.0, 255.0).astype(np.uint8)
    else:
        crop = np.asarray(crop, dtype=np.uint8)
    return np.ascontiguousarray(crop.transpose(2, 0, 1))


def close_hires_wrist_renderer(task) -> None:
    for attr in ("_hires_wrist_renderer", "_hires_front_renderer", "_hires_left_renderer"):
        rend = getattr(task, attr, None)
        if rend is not None:
            try:
                rend.close()
            except Exception:
                pass
            setattr(task, attr, None)


def render_front_hires_96(task, subtask: str) -> np.ndarray:
    """Native MuJoCo 640x640 -> 96x96 centered front stereo ROI (~1.33 px/mm at dock, zero privilege)."""
    rend = getattr(task, "_hires_front_renderer", None)
    if rend is None:
        rend = mujoco.Renderer(task.m, 640, 640)
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        task._hires_front_renderer = rend
        task._hires_front_opt = opt
    rend.update_scene(task.d, camera="front", scene_option=task._hires_front_opt)
    raw = rend.render()
    if subtask == "p4_insert":
        crop = raw[308:404, 218:314, :]  # centered at (u=266, v=356), +/-37mm around port
    else:
        crop = raw[272:368, 360:456, :]  # centered at (u=408, v=320), +/-38mm around holder slot
    br = float(getattr(task, "brightness", 1.0))
    if br != 1.0:
        crop = np.clip(crop.astype(np.float32) * br, 0.0, 255.0).astype(np.uint8)
    else:
        crop = np.asarray(crop, dtype=np.uint8)
    return np.ascontiguousarray(crop.transpose(2, 0, 1))


def render_left_hires_96(task, subtask: str) -> np.ndarray:
    """Native MuJoCo 640x640 -> 96x96 orthogonal side-profile left_wrist ROI (~1.02 px/mm, zero privilege).

    left_wrist views purely horizontally along -S[:,0] (83 deg orthogonal to wrist, 78 deg to front),
    giving direct sub-millimeter observability of insertion depth dep_z and pitch angle rx.
    """
    rend = getattr(task, "_hires_left_renderer", None)
    if rend is None:
        rend = mujoco.Renderer(task.m, 640, 640)
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        task._hires_left_renderer = rend
        task._hires_left_opt = opt
    rend.update_scene(task.d, camera="left_wrist", scene_option=task._hires_left_opt)
    raw = rend.render()
    if subtask == "p4_insert":
        crop = raw[346:442, 105:201, :]  # centered at (u=153, v=394), covers -36mm hover to +12mm bottom
    else:
        crop = raw[325:421, 148:244, :]  # centered at (u=196, v=373), covers holder slot (+/-63mm)
    br = float(getattr(task, "brightness", 1.0))
    if br != 1.0:
        crop = np.clip(crop.astype(np.float32) * br, 0.0, 255.0).astype(np.uint8)
    else:
        crop = np.asarray(crop, dtype=np.uint8)
    return np.ascontiguousarray(crop.transpose(2, 0, 1))


def extract_wrist_128(task, obs: dict, *, hires_wrist: bool = True) -> np.ndarray:
    if hires_wrist:
        return render_wrist_hires_128(task)
    return crop_wrist_128(obs["observation.images.wrist"])


def extract_front_96(task, obs: dict, subtask: str, *, hires_front: bool = False) -> np.ndarray:
    if hires_front:
        return render_front_hires_96(task, subtask)
    return crop_front_96(obs["observation.images.front"], subtask)


def extract_left_96(task, obs: dict, subtask: str, *, hires_left: bool = False) -> np.ndarray | None:
    if hires_left:
        return render_left_hires_96(task, subtask)
    return None


def crop_front_96(front_rgb_224: np.ndarray, subtask: str) -> np.ndarray:
    """Orthogonal 96x96 stereo crop from front camera -> uint8 CHW (3, 96, 96)."""
    img = np.asarray(front_rgb_224, dtype=np.uint8)
    if subtask == "p4_insert":
        crop = img[76:172, 48:144, :]  # dock centered around (u=96, v=124)
    else:
        crop = img[64:160, 96:192, :]  # holder centered around (u=142, v=112)
    return np.ascontiguousarray(crop.transpose(2, 0, 1))


class SpatialSoftmaxEncoder(nn.Module):
    """High-res SpatialSoftmax CNN (stride 2->2->1: 32x32 for 128x128, 24x24 for 96x96)."""

    def __init__(self, in_h: int = 128, in_w: int = 128, out_dim: int = 64, num_kp: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, num_kp, kernel_size=1),
        )
        fh, fw = in_h // 4, in_w // 4
        xs = torch.linspace(-1.0, 1.0, fw)
        ys = torch.linspace(-1.0, 1.0, fh)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("gx", gx.reshape(1, 1, fh * fw))
        self.register_buffer("gy", gy.reshape(1, 1, fh * fw))
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.proj = nn.Sequential(
            nn.Linear(num_kp * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, img_u8_or_f: torch.Tensor, *, augment: bool = False, shift_aug: bool = True) -> torch.Tensor:
        x = img_u8_or_f.float() / 255.0
        if augment and self.training:
            b, c, h, w = x.shape
            if shift_aug:
                # Replicate-padded random shift (+/- 2 px)
                pad = 2
                xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
                dy = int(torch.randint(0, 2 * pad + 1, (1,)).item())
                dx = int(torch.randint(0, 2 * pad + 1, (1,)).item())
                x = xp[:, :, dy : dy + h, dx : dx + w]
            if os.environ.get("STAGE2_GENTLE_PHOTO_AUG", "0") == "1":
                gain = 1.0 + (torch.rand((b, 1, 1, 1), device=x.device) - 0.5) * 0.16
                bias = (torch.rand((b, 1, 1, 1), device=x.device) - 0.5) * 0.04
                x = torch.clamp(x * gain + bias, 0.0, 1.0)
            else:
                gain = 1.0 + (torch.rand((b, 1, 1, 1), device=x.device) - 0.5) * 0.56
                ch_gain = 1.0 + (torch.rand((b, c, 1, 1), device=x.device) - 0.5) * 0.10
                bias = (torch.rand((b, 1, 1, 1), device=x.device) - 0.5) * 0.16
                noise = torch.randn_like(x) * 0.012
                x = torch.clamp(x * gain * ch_gain + bias + noise, 0.0, 1.0)
        feat = self.conv(x)  # (B, K, fh, fw)
        b, k, fh, fw = feat.shape
        temp = torch.exp(self.log_temp.clamp(-2.0, 2.0))
        prob = F.softmax(feat.reshape(b, k, fh * fw) / temp, dim=-1)
        ex = (prob * self.gx).sum(dim=-1)
        ey = (prob * self.gy).sum(dim=-1)
        kp = torch.cat([ex, ey], dim=-1)
        return self.proj(kp)


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, pre_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        if pre_norm:
            layers.append(nn.LayerNorm(in_dim))
        layers.extend([nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CartesianResidualActor(nn.Module):
    """Strictly robot-only stereo/tri-stereo visual + contact Actor outputting bounded 6D Cartesian step [-1, 1]^6."""

    def __init__(
        self,
        z_dim: int = 2048,
        proprio_dim: int = rcu.CONTACT_PROPRIO_DIM,
        ref_dim: int = 32,
        action_dim: int = 6,
        modality_dim: int = 64,
        hidden_dim: int = 256,
        use_left_wrist: bool = False,
    ):
        super().__init__()
        self.use_left_wrist = bool(use_left_wrist)
        self.z_enc = ModalityEncoder(z_dim, modality_dim, pre_norm=True)
        self.p_enc = ModalityEncoder(proprio_dim, modality_dim, pre_norm=True)
        self.r_enc = ModalityEncoder(ref_dim, modality_dim, pre_norm=False)
        self.w_enc = SpatialSoftmaxEncoder(in_h=128, in_w=128, out_dim=modality_dim, num_kp=32)
        self.f_enc = SpatialSoftmaxEncoder(in_h=96, in_w=96, out_dim=modality_dim, num_kp=32)
        if self.use_left_wrist:
            self.l_enc = SpatialSoftmaxEncoder(in_h=96, in_w=96, out_dim=modality_dim, num_kp=32)
        n_mod = 6 if self.use_left_wrist else 5
        self.backbone = nn.Sequential(
            nn.Linear(modality_dim * n_mod, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.out = nn.Linear(hidden_dim, action_dim)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out.bias)
        # Training-ONLY auxiliary alignment regression head (NEVER read or called at inference/eval/test!)
        self.aux_geom_head = nn.Linear(hidden_dim, 6)

    def encode_trunk(
        self,
        z_rl: torch.Tensor,
        proprio: torch.Tensor,
        ref_rel32: torch.Tensor,
        wrist_crop: torch.Tensor,
        front_crop: torch.Tensor,
        left_crop: torch.Tensor | None = None,
        *,
        ref_dropout_prob: float = 0.0,
        proprio_dropout_prob: float = 0.0,
        prev_act_drop_prob: float = 0.0,
        augment: bool = False,
        shift_aug: bool = True,
    ) -> torch.Tensor:
        ref = ref_rel32
        prop = proprio
        if self.training:
            if ref_dropout_prob > 0.0:
                keep = (torch.rand((ref.shape[0], 1), device=ref.device) >= ref_dropout_prob).to(ref.dtype)
                ref = ref * keep
            if prev_act_drop_prob > 0.0 and prop.shape[-1] >= 44:
                keep_pa = (torch.rand((prop.shape[0], 1), device=prop.device) >= prev_act_drop_prob).to(prop.dtype)
                prop = prop.clone()
                prop[:, 38:44] = prop[:, 38:44] * keep_pa
            if proprio_dropout_prob > 0.0:
                keep_p = (torch.rand((prop.shape[0], 1), device=prop.device) >= proprio_dropout_prob).to(prop.dtype)
                prop = prop * keep_p
        feats = [
            self.z_enc(z_rl),
            self.p_enc(prop),
            self.r_enc(ref),
            self.w_enc(wrist_crop, augment=augment, shift_aug=shift_aug),
            self.f_enc(front_crop, augment=augment, shift_aug=shift_aug),
        ]
        if self.use_left_wrist:
            assert left_crop is not None, "left_crop required when use_left_wrist=True"
            feats.append(self.l_enc(left_crop, augment=augment, shift_aug=shift_aug))
        h = torch.cat(feats, dim=-1)
        return self.backbone(h)

    def forward(
        self,
        z_rl: torch.Tensor,
        proprio: torch.Tensor,
        ref_rel32: torch.Tensor,
        wrist_crop: torch.Tensor,
        front_crop: torch.Tensor,
        left_crop: torch.Tensor | None = None,
        *,
        ref_dropout_prob: float = 0.0,
        proprio_dropout_prob: float = 0.0,
        prev_act_drop_prob: float = 0.0,
        augment: bool = False,
        shift_aug: bool = True,
        return_aux: bool = False,
    ):
        feat = self.encode_trunk(
            z_rl, proprio, ref_rel32, wrist_crop, front_crop, left_crop,
            ref_dropout_prob=ref_dropout_prob,
            proprio_dropout_prob=proprio_dropout_prob,
            prev_act_drop_prob=prev_act_drop_prob,
            augment=augment,
            shift_aug=shift_aug,
        )
        act = torch.tanh(self.out(feat))
        if return_aux:
            return act, self.aux_geom_head(feat)
        return act


class ActionSkipQHead(nn.Module):
    """Training-only Asymmetric Action-Skip Q-head (uses geom6 only during sim training for low-variance Q)."""

    def __init__(
        self,
        z_dim: int = 2048,
        proprio_dim: int = rcu.CONTACT_PROPRIO_DIM,
        ref_dim: int = 32,
        geom_dim: int = 6,
        action_dim: int = 6,
        modality_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.z_enc = ModalityEncoder(z_dim, modality_dim, pre_norm=True)
        self.p_enc = ModalityEncoder(proprio_dim, modality_dim, pre_norm=True)
        self.r_enc = ModalityEncoder(ref_dim, modality_dim, pre_norm=False)
        self.g_enc = ModalityEncoder(geom_dim, modality_dim, pre_norm=False)
        self.a_enc = ModalityEncoder(action_dim, modality_dim, pre_norm=False)
        self.fc1 = nn.Sequential(
            nn.Linear(modality_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim + modality_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.out = nn.Linear(hidden_dim + modality_dim, 1)

    def forward(
        self,
        z_rl: torch.Tensor,
        proprio: torch.Tensor,
        ref_rel32: torch.Tensor,
        geom6: torch.Tensor,
        action6: torch.Tensor,
    ) -> torch.Tensor:
        hz = self.z_enc(z_rl)
        hp = self.p_enc(proprio)
        hr = self.r_enc(ref_rel32)
        hg = self.g_enc(geom6)
        ha = self.a_enc(action6)
        h1 = self.fc1(torch.cat([hz, hp, hr, hg, ha], dim=-1))
        h2 = self.fc2(torch.cat([h1, ha], dim=-1))
        return self.out(torch.cat([h2, ha], dim=-1))


class TwinActionSkipCritic(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.q1 = ActionSkipQHead(**kwargs)
        self.q2 = ActionSkipQHead(**kwargs)

    def forward(
        self,
        z_rl: torch.Tensor,
        proprio: torch.Tensor,
        ref_rel32: torch.Tensor,
        geom6: torch.Tensor,
        action6: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1(z_rl, proprio, ref_rel32, geom6, action6),
            self.q2(z_rl, proprio, ref_rel32, geom6, action6),
        )


class Stage1RLTFeatureProvider:
    """Frozen Stage 1 pi0.5 + RLTTokenTransformer providing (z_rl, ref_abs_chunk4, ref_rel32)."""

    def __init__(self, config_dir: Path, config_name: str, checkpoint: str, device: torch.device, rlt_adapter_pt: str = ""):
        usb12.register()
        self.device = device
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name=config_name)
        inp, out = build_openpi_transforms(
            cfg.actor.model.model_path,
            cfg.actor.model.openpi.config_name,
            dict(cfg.actor.model.openpi_data),
        )
        self.tx, self.un = T.compose(inp), T.compose(out)
        self.vla = get_model(cfg.actor.model, torch_dtype=torch.float32).eval()
        state = {"fsdp_checkpoint": {"model": self.vla.state_dict()}}
        dcp.load(state, checkpoint_id=checkpoint)
        self.vla.load_state_dict(state["fsdp_checkpoint"]["model"], strict=True)
        del state
        for p in self.vla.parameters():
            p.requires_grad = False
        self.vla = self.vla.to(device)
        if rlt_adapter_pt:
            ck = torch.load(rlt_adapter_pt, map_location="cpu")
            self.rlt = RLTTokenTransformer(**ck["rlt_cfg"]).eval().to(device)
            self.rlt.load_state_dict(ck["rlt_module"], strict=True)
            for p in self.rlt.parameters():
                p.requires_grad = False
        elif hasattr(self.vla, "rlt_module"):
            self.rlt = self.vla.rlt_module.eval()
        else:
            self.rlt = None

    @torch.inference_mode()
    def infer(
        self, obs: dict, q_home0: np.ndarray, phase_idx: int, step_seed: int, *, flow_m: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        st21 = usb12.make_state21(obs["observation.state"], q_home0)
        raw = dict(
            state=st21,
            image=obs["observation.images.front"],
            left_wrist_image=obs["observation.images.left_wrist"],
            wrist_image=obs["observation.images.wrist"],
            prompt=usb12.PHASE_PROMPTS[phase_idx],
        )
        proc = self.tx(raw)
        batch = self.vla._observation_to_device(
            default_collate([{k: v for k, v in proc.items() if k != "actions"}])
        )
        torch.manual_seed(step_seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            obs_prep = openpi_model.preprocess_observation(batch, train=False)
            prefix_out, prefix_mask, kv_cache = self.vla.build_prefix_cache(obs_prep)
            z_rl = (
                self.rlt.encode_flat(prefix_out.float(), prefix_mask)[0].float().cpu().numpy()
                if self.rlt is not None
                else None
            )

            M = int(flow_m)
            kv_exp = tuple(tuple(t.expand(M, *t.shape[1:]) for t in layer) for layer in kv_cache)
            pm_exp = prefix_mask.expand(M, -1)
            obs_k = dataclasses.replace(obs_prep, state=obs_prep.state.expand(M, -1))
            x_t = torch.randn((M, 16, 32), device=self.device, dtype=torch.float32)
            dt = -1.0 / 5.0
            t_val = 1.0
            while t_val >= -dt / 2:
                t_tensor = torch.full((M,), t_val, device=self.device, dtype=torch.float32)
                sout = self.vla.run_suffix(obs_k, x_t, t_tensor, kv_exp, pm_exp)
                vt = self.vla.velocity_from_suffix(sout)
                x_t = x_t + dt * vt.float()
                t_val += dt
            pred_norm = x_t.mean(dim=0).float().cpu().numpy()
        ref_rel32 = pred_norm[:4, :8].reshape(32).astype(np.float32)
        abs16 = self.un(dict(actions=pred_norm, state=proc["state"]))["actions"]
        return (z_rl.astype(np.float32) if z_rl is not None else None), abs16[:4, :8].astype(np.float32), ref_rel32


def step_persistent_qtgt_dq(
    task,
    act6: np.ndarray,
    *,
    scale_xyz_m: float = 0.0018,
    scale_rot_rad: float = 0.022,
    max_dq_rad: float = 0.025,
    damping: float = 0.02,
    nullspace_gain: float = 0.02,
    exact_nullspace: bool = False,
) -> np.ndarray:
    """Strictly robot-only mapping from act6 in [-1, 1]^6 to right-arm dq via URDF Jacobian + nullspace limit avoidance."""
    a = np.clip(np.asarray(act6, dtype=np.float64).reshape(6), -1.0, 1.0)
    dx_w = np.empty(6, dtype=np.float64)
    dx_w[:3] = a[:3] * float(scale_xyz_m)
    dx_w[3:] = a[3:] * float(scale_rot_rad)
    J = rcu.compute_robot_tcp_jacobian(task)
    eff_damp = 0.008 if exact_nullspace else damping
    JJt_inv = np.linalg.inv(J @ J.T + (eff_damp ** 2) * np.eye(6))
    J_pinv = J.T @ JJt_inv
    dq = J_pinv @ dx_w
    if exact_nullspace:
        q_cur = task.rd.qpos[rcu.ARM_DOF_SLICE]
        j_ids = [int(task.rm.dof_jntid[d]) for d in range(24, 31)]
        push = np.zeros(7, dtype=np.float64)
        for k, j in enumerate(j_ids):
            lo, hi = task.rm.jnt_range[j, 0], task.rm.jnt_range[j, 1]
            if (q_cur[k] - lo) < 0.12:
                push[k] = 0.12 - (q_cur[k] - lo)
            elif (hi - q_cur[k]) < 0.12:
                push[k] = -(0.12 - (hi - q_cur[k]))
        if np.any(push != 0.0):
            N_exact = np.eye(7) - np.linalg.pinv(J, rcond=1e-4) @ J
            dq = dq + 0.15 * (N_exact @ push)
    elif nullspace_gain > 0.0:
        # Robot-only URDF joint-midrange nullspace bias prevents wrist joint-limit stalls
        q_cur = task.rd.qpos[rcu.ARM_DOF_SLICE]
        j_ids = [int(task.rm.dof_jntid[d]) for d in range(24, 31)]
        q_mid = np.array([0.5 * (task.rm.jnt_range[j, 0] + task.rm.jnt_range[j, 1]) for j in j_ids], dtype=np.float64)
        N = np.eye(7) - J_pinv @ J
        dq = dq + nullspace_gain * (N @ (q_mid - q_cur))
    return np.clip(dq, -max_dq_rad, max_dq_rad).astype(np.float64)


class CompactDemoBuffer:
    """Zero-copy indexed offline RLPD demo buffer loaded from collect_rlt_stage2_buffer.py .pt file."""

    def __init__(self, pt_path: str | Path):
        d = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        self.z_table = d["z_table"]         # (V, 2048) fp16
        self.ref_table = d["ref_table"]     # (V, 32) fp32
        self.vla_idx = d["vla_idx"]
        self.nvla_idx = d["nvla_idx"]
        self.p = d["p"]
        self.w = d["w"]
        self.f = d["f"]
        self.l = d.get("l", None)
        self.g = d["g"]
        self.a = d["a"]
        self.r = d["r"]
        self.np_ = d["np_"]
        self.nw = d["nw"]
        self.nf = d["nf"]
        self.nl = d.get("nl", None)
        self.ng = d["ng"]
        self.done = d["done"]
        self.mc = d["mc"]
        self.ep_succ = d["ep_succ"]
        self.is_expert = d["is_expert"]
        self.succ_indices = np.nonzero(self.ep_succ)[0]
        self.size = int(len(self.r))

    def sample(self, n: int, rng: np.random.Generator, *, only_succ: bool = False) -> dict[str, np.ndarray]:
        if only_succ and len(self.succ_indices) > 0:
            idx = self.succ_indices[rng.integers(0, len(self.succ_indices), size=n)]
        else:
            idx = rng.integers(0, self.size, size=n)
        out = dict(
            z=self.z_table[self.vla_idx[idx]].astype(np.float32),
            p=self.p[idx],
            ref=self.ref_table[self.vla_idx[idx]],
            w=self.w[idx],
            f=self.f[idx],
            g=self.g[idx],
            a=self.a[idx],
            r=self.r[idx][:, None],
            nz=self.z_table[self.nvla_idx[idx]].astype(np.float32),
            np_=self.np_[idx],
            nref=self.ref_table[self.nvla_idx[idx]],
            nw=self.nw[idx],
            nf=self.nf[idx],
            ng=self.ng[idx],
            done=self.done[idx][:, None],
            mc=self.mc[idx][:, None],
            succ=self.ep_succ[idx].astype(np.float32)[:, None],
            exp=self.is_expert[idx].astype(np.float32)[:, None],
        )
        if self.l is not None and self.nl is not None:
            out["l"] = self.l[idx]
            out["nl"] = self.nl[idx]
        return out


class ResidualRLPDTrainer:
    """Off-policy Asymmetric RLPD (Twin-Q TD3 + BC-human/HG-DAgger + AWR + Aux-Geom + Critic Ranking Gate)."""

    def __init__(
        self,
        device: torch.device,
        *,
        gamma: float = 0.98,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        bc_reg_weight: float = 0.002,
        bc_human_weight: float = 2.5,
        aux_geom_weight: float = 0.6,
        awr_weight: float = 1.0,
        awr_beta: float = 2.0,
        q_weight_max: float = 0.30,
        ref_dropout_prob: float = 0.20,
        proprio_dropout_prob: float = 0.0,
        prev_act_drop_prob: float = 0.0,
        aux_xyrot_boost: bool = False,
        critic_actor_ratio: int = 2,
        shift_aug: bool = True,
        use_left_wrist: bool = False,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.bc_reg_weight = bc_reg_weight
        self.bc_human_weight = bc_human_weight
        self.aux_geom_weight = aux_geom_weight
        self.awr_weight = awr_weight
        self.awr_beta = awr_beta
        self.q_weight_max = q_weight_max
        self.ref_dropout_prob = ref_dropout_prob
        self.proprio_dropout_prob = proprio_dropout_prob
        self.prev_act_drop_prob = prev_act_drop_prob
        self.aux_xyrot_boost = bool(aux_xyrot_boost)
        self.critic_actor_ratio = critic_actor_ratio
        self.shift_aug = shift_aug
        self.use_left_wrist = bool(use_left_wrist)

        self.actor = CartesianResidualActor(use_left_wrist=self.use_left_wrist).to(device)
        self.target_actor = copy.deepcopy(self.actor).eval().to(device)
        self.critic = TwinActionSkipCritic().to(device)
        self.target_critic = copy.deepcopy(self.critic).eval().to(device)
        for p in list(self.target_actor.parameters()) + list(self.target_critic.parameters()):
            p.requires_grad = False

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=1e-4)
        self.critic_gate_passed = False
        self.updates_since_gate = 0
        self.total_updates = 0

    def _pack_list(self, items: list[dict]) -> dict[str, np.ndarray]:
        out = dict(
            z=np.stack([x["z"] for x in items]).astype(np.float32),
            p=np.stack([x["p"] for x in items]).astype(np.float32),
            ref=np.stack([x["ref"] for x in items]).astype(np.float32),
            w=np.stack([x["w"] for x in items]).astype(np.uint8),
            f=np.stack([x["f"] for x in items]).astype(np.uint8),
            g=np.stack([x["g"] for x in items]).astype(np.float32),
            a=np.stack([x["a"] for x in items]).astype(np.float32),
            r=np.array([[x["r"]] for x in items], dtype=np.float32),
            nz=np.stack([x["nz"] for x in items]).astype(np.float32),
            np_=np.stack([x["np"] for x in items]).astype(np.float32),
            nref=np.stack([x["nref"] for x in items]).astype(np.float32),
            nw=np.stack([x["nw"] for x in items]).astype(np.uint8),
            nf=np.stack([x["nf"] for x in items]).astype(np.uint8),
            ng=np.stack([x["ng"] for x in items]).astype(np.float32),
            done=np.array([[x["done"]] for x in items], dtype=np.float32),
            mc=np.array([[x["mc"]] for x in items], dtype=np.float32),
            succ=np.array([[1.0 if x["ep_succ"] else 0.0] for x in items], dtype=np.float32),
            exp=np.array([[1.0 if x["is_expert"] else 0.0] for x in items], dtype=np.float32),
        )
        if self.use_left_wrist:
            out["l"] = np.stack([x["l"] for x in items]).astype(np.uint8)
            out["nl"] = np.stack([x["nl"] for x in items]).astype(np.uint8)
        return out

    def sample_batch(
        self,
        all_buf: list[dict],
        succ_buf: list[dict],
        batch_size: int,
        rng: np.random.Generator,
        *,
        demo_buf: CompactDemoBuffer | None = None,
    ) -> dict[str, torch.Tensor]:
        parts: list[dict[str, np.ndarray]] = []
        if demo_buf is not None and len(all_buf) < 64:
            # Pure demo buffer pretraining (70% successful + 30% all for negative Q contrast)
            n_s = int(batch_size * 0.75)
            parts.append(demo_buf.sample(n_s, rng, only_succ=True))
            parts.append(demo_buf.sample(batch_size - n_s, rng, only_succ=False))
        elif demo_buf is not None:
            # Official RLPD 50% offline demo buffer + 50% online buffer (w/ 50% online succ)
            n_demo = batch_size // 2
            n_on = batch_size - n_demo
            parts.append(demo_buf.sample(n_demo, rng, only_succ=False))
            if len(succ_buf) >= 16:
                n_os = n_on // 2
                items = [succ_buf[i] for i in rng.integers(0, len(succ_buf), size=n_os)] + [
                    all_buf[i] for i in rng.integers(0, len(all_buf), size=n_on - n_os)
                ]
            else:
                items = [all_buf[i] for i in rng.integers(0, len(all_buf), size=n_on)]
            parts.append(self._pack_list(items))
        else:
            if len(succ_buf) >= 16:
                n_s = batch_size // 2
                items = [succ_buf[i] for i in rng.integers(0, len(succ_buf), size=n_s)] + [
                    all_buf[i] for i in rng.integers(0, len(all_buf), size=batch_size - n_s)
                ]
            else:
                items = [all_buf[i] for i in rng.integers(0, len(all_buf), size=batch_size)]
            parts.append(self._pack_list(items))

        merged = {k: np.concatenate([pt[k] for pt in parts], axis=0) for k in parts[0]}
        return {k: torch.from_numpy(v).to(self.device) for k, v in merged.items()}

    def update(
        self,
        all_buf: list[dict],
        succ_buf: list[dict],
        rng: np.random.Generator,
        steps: int = 350,
        batch_size: int = 128,
        *,
        demo_buf: CompactDemoBuffer | None = None,
        allow_q_grad: bool = True,
    ) -> dict:
        self.actor.train()
        self.critic.train()
        last_m = {}
        a_loss_val = bc_h_val = aux_val = q_pi_val = q_w = 0.0
        for _ in range(steps):
            b = self.sample_batch(all_buf, succ_buf, batch_size, rng, demo_buf=demo_buf)
            with torch.no_grad():
                na = self.target_actor(b["nz"], b["np_"], b["nref"], b["nw"], b["nf"], b.get("nl"))
                noise = torch.clamp(torch.randn_like(na) * 0.05, -0.12, 0.12)
                na = torch.clamp(na + noise, -1.0, 1.0)
                tq1, tq2 = self.target_critic(b["nz"], b["np_"], b["nref"], b["ng"], na)
                td_target = b["r"] + (1.0 - b["done"]) * self.gamma * torch.minimum(tq1, tq2)
                target_q = 0.80 * td_target + 0.20 * b["mc"]

            q1, q2 = self.critic(b["z"], b["p"], b["ref"], b["g"], b["a"])
            c_loss = F.smooth_l1_loss(q1, target_q, beta=2.0) + F.smooth_l1_loss(q2, target_q, beta=2.0)
            self.critic_opt.zero_grad(set_to_none=True)
            c_loss.backward()
            c_gn = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0))
            self.critic_opt.step()

            self.total_updates += 1
            if self.total_updates % self.critic_actor_ratio == 0:
                pred_a, pred_g = self.actor(
                    b["z"], b["p"], b["ref"], b["w"], b["f"], b.get("l"),
                    ref_dropout_prob=self.ref_dropout_prob,
                    proprio_dropout_prob=self.proprio_dropout_prob,
                    prev_act_drop_prob=self.prev_act_drop_prob,
                    augment=True,
                    shift_aug=self.shift_aug,
                    return_aux=True,
                )
                reg_loss = pred_a.pow(2).mean()
                if self.aux_xyrot_boost:
                    w_g = torch.tensor([4.0, 4.0, 1.0, 4.0, 4.0, 2.0], device=self.device, dtype=pred_g.dtype)
                    aux_loss = (F.smooth_l1_loss(pred_g, b["g"], reduction="none") * w_g).mean()
                else:
                    aux_loss = F.smooth_l1_loss(pred_g, b["g"])
                aux_val = float(aux_loss.item())

                # RLinf BC-human / HG-DAgger intervention loss on SUCCESSFUL expert transitions only
                exp_mask = (b["exp"].squeeze(-1) > 0.5) & (b["succ"].squeeze(-1) > 0.5)
                if exp_mask.any():
                    bc_h = F.mse_loss(pred_a[exp_mask], b["a"][exp_mask]) + 0.5 * F.smooth_l1_loss(pred_a[exp_mask], b["a"][exp_mask])
                    bc_h_val = float(bc_h.item())
                else:
                    bc_h = torch.zeros((), device=self.device)

                # AWR on ALL successful episodes (autonomous + expert)
                succ_mask = b["succ"].squeeze(-1) > 0.5
                if succ_mask.any():
                    with torch.no_grad():
                        qa1, qa2 = self.critic(b["z"][succ_mask], b["p"][succ_mask], b["ref"][succ_mask], b["g"][succ_mask], b["a"][succ_mask])
                        q_data = torch.minimum(qa1, qa2).squeeze(-1)
                        adv = q_data - q_data.mean()
                        w_awr = torch.exp((adv / max(self.awr_beta, 1e-3)).clamp(-3.0, 3.0))
                        w_awr = w_awr / w_awr.mean().clamp_min(1e-4)
                    per_mse = (pred_a[succ_mask] - b["a"][succ_mask]).pow(2).mean(dim=-1)
                    awr_loss = (w_awr * per_mse).mean()
                else:
                    awr_loss = torch.zeros((), device=self.device)

                q_loss = torch.zeros((), device=self.device)
                if allow_q_grad and self.critic_gate_passed:
                    self.updates_since_gate += 1
                    ramp = min(1.0, self.updates_since_gate / 600.0)
                    q_w = self.q_weight_max * ramp
                    qp1, qp2 = self.critic(b["z"], b["p"], b["ref"], b["g"], pred_a)
                    q_min = torch.minimum(qp1, qp2)
                    q_pi_val = float(q_min.mean().item())
                    q_scale = q_min.detach().abs().mean().clamp_min(1.0)
                    q_loss = -(q_min / q_scale).mean()

                a_loss = (
                    q_w * q_loss
                    + self.bc_human_weight * bc_h
                    + self.awr_weight * awr_loss
                    + self.aux_geom_weight * aux_loss
                    + self.bc_reg_weight * reg_loss
                )
                self.actor_opt.zero_grad(set_to_none=True)
                a_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
                self.actor_opt.step()
                a_loss_val = float(a_loss.item())

                with torch.no_grad():
                    for tp, sp in zip(self.target_actor.parameters(), self.actor.parameters()):
                        tp.data.lerp_(sp.data, self.tau)

            with torch.no_grad():
                for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
                    tp.data.lerp_(sp.data, self.tau)

            last_m = dict(
                c_loss=round(float(c_loss.item()), 4),
                q1_mean=round(float(q1.mean().item()), 3),
                c_gn=round(c_gn, 3),
                a_loss=round(a_loss_val, 4),
                bc_h=round(bc_h_val, 4),
                aux_g=round(aux_val, 4),
                q_pi=round(q_pi_val, 3),
                q_w=round(q_w, 4),
                gate_open=self.critic_gate_passed,
            )
        return last_m

    @torch.no_grad()
    def eval_critic_gate(self, buf: list[dict]) -> dict:
        if len(buf) < 64:
            return dict(corr=0.0, gap=0.0, passed=False)
        self.critic.eval()
        sub = buf[-min(len(buf), 1024):]
        q_all = []
        for s in range(0, len(sub), 128):
            sl = sub[s : s + 128]
            z = torch.from_numpy(np.stack([x["z"] for x in sl])).to(self.device)
            p = torch.from_numpy(np.stack([x["p"] for x in sl])).to(self.device)
            ref = torch.from_numpy(np.stack([x["ref"] for x in sl])).to(self.device)
            g = torch.from_numpy(np.stack([x["g"] for x in sl])).to(self.device)
            a = torch.from_numpy(np.stack([x["a"] for x in sl])).to(self.device)
            q1, q2 = self.critic(z, p, ref, g, a)
            q_all.append(torch.minimum(q1, q2).squeeze(-1).cpu().numpy())
        q = np.concatenate(q_all)
        mc = np.array([x["mc"] for x in sub], dtype=np.float32)
        ok = np.array([x["ep_succ"] for x in sub], dtype=bool)
        corr = float(np.corrcoef(q, mc)[0, 1]) if np.std(q) > 1e-8 and np.std(mc) > 1e-8 else 0.0
        sp = float(spearmanr(q, mc).statistic) if np.std(q) > 1e-8 and np.std(mc) > 1e-8 else 0.0
        if not np.isfinite(sp):
            sp = 0.0
        gap = float(q[ok].mean() - q[~ok].mean()) if ok.any() and (~ok).any() else 0.0
        passed = bool(corr >= 0.45 and sp >= 0.50 and gap >= 0.8)
        if passed:
            self.critic_gate_passed = True
        return dict(corr=round(corr, 4), spearman=round(sp, 4), gap=round(gap, 4), passed=self.critic_gate_passed)


def compute_train_only_geom6(task, subtask: str) -> np.ndarray:
    """Training-only normalized 6D alignment vector for auxiliary visual supervision & asymmetric critic."""
    tip, Rp = task.object_pose()
    if subtask == "p4_insert":
        local_m = task.S.T @ (tip - task.port)
        rotvec_err = Rotation.from_matrix(task.S @ Rp.T).as_rotvec()
        g = np.concatenate([
            np.clip(local_m * 100.0, -3.0, 3.0),      # 10 mm -> 1.0
            np.clip(rotvec_err * 10.0, -3.0, 3.0),    # 0.1 rad (~5.7 deg) -> 1.0
        ])
    else:
        delta_m = tip - task.initial_tip
        rotvec_err = Rotation.from_matrix(task.initial_rotation @ Rp.T).as_rotvec()
        g = np.concatenate([
            np.clip(delta_m * 50.0, -3.0, 3.0),       # 20 mm -> 1.0
            np.clip(rotvec_err * 10.0, -3.0, 3.0),
        ])
    return g.astype(np.float32)


def compute_subtask_potential(task, subtask: str) -> tuple[float, dict]:
    tip, Rp = task.object_pose()
    if subtask == "p4_insert":
        met = task.insertion_metrics()
        dep_mm = float(met["depth_m"] * 1000.0)
        lat_mm = float(met["lateral_error_m"] * 1000.0)
        ang_deg = float(np.degrees(met["angle_error_rad"]))
        pot = -0.45 * min(lat_mm, 15.0) + 0.70 * float(np.clip(dep_mm, -15.0, 12.0)) - 0.18 * min(ang_deg, 15.0)
        return pot, dict(lat_mm=lat_mm, dep_mm=dep_mm, ang_deg=ang_deg)
    else:
        delta = tip - task.initial_tip
        hxy_mm = float(np.linalg.norm(delta[:2]) * 1000.0)
        hdz_mm = float(abs(delta[2]) * 1000.0)
        pot = -0.45 * min(hxy_mm, 20.0) - 0.55 * min(hdz_mm, 30.0)
        return pot, dict(hxy_mm=hxy_mm, hdz_mm=hdz_mm)


def compute_train_only_expert_act6(
    task,
    subtask: str,
    *,
    scale_xyz_m: float = 0.0018,
    scale_rot_rad: float = 0.022,
    compliant_p4: bool = False,
    noback_p4: bool = False,
    soft_p7: bool = False,
    smooth_p7: bool = False,
) -> np.ndarray:
    """Training-only simulator expert in exact [-1, 1]^6 persistent q_tgt action space (NEVER used in eval/test!)."""
    tip, Rp = task.object_pose()
    task.robot_forward()
    p_tcp = task.rd.site_xpos[task.rtool].copy()
    r_lever = tip - p_tcp

    if subtask == "p4_insert":
        S = task.S
        met = task.insertion_metrics()
        dep = met["depth_m"]
        lat = met["lateral_error_m"]
        dp = tip - task.port
        lat_vec = dp - dep * S[:, 2]
        rotvec_err = Rotation.from_matrix(S @ Rp.T).as_rotvec()
        ang_rad = float(np.linalg.norm(rotvec_err))
        if noback_p4:
            # Compliant never-back-out v3_real expert: targets 11.15mm (>=10.5mm pass, <12.6mm floor) with dz>=0 inside socket
            if dep >= 0.0106 and lat <= 0.00065:
                dz_push = max(0.0, 0.22 * (0.01115 - dep))
                dp_tip = -0.25 * lat_vec + dz_push * S[:, 2]
                drot_w = np.clip(0.28 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            elif dep >= 0.0012 and lat <= 0.0010:
                dz_push = max(0.0, (0.35 if ang_rad <= 0.038 else 0.10) * (0.01115 - dep))
                dp_tip = -0.50 * lat_vec + dz_push * S[:, 2]
                drot_w = np.clip(0.55 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            elif lat <= 0.00045 and ang_rad <= 0.022:
                dp_tip = -0.65 * lat_vec + 0.42 * (0.01115 - dep) * S[:, 2]
                drot_w = np.clip(0.55 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            else:
                align_err = max(lat / 0.00045, ang_rad / 0.022)
                tgt_dep = -float(np.clip(0.0015 + 0.0012 * (align_err - 1.0), 0.0015, 0.0045))
                dp_tip = -0.78 * lat_vec + 0.45 * (tgt_dep - dep) * S[:, 2]
                drot_w = np.clip(0.55 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            dp_tcp = np.clip(dp_tip - np.cross(drot_w, r_lever), -scale_xyz_m, scale_xyz_m)
        elif compliant_p4:
            # Compliant v3_real expert: never yanks out inside socket, pauses push if tilted, soft-holds at 10.9mm (<25N)
            if dep >= 0.0106 and lat <= 0.0006:
                dp_tip = -0.25 * lat_vec + 0.15 * (0.0109 - dep) * S[:, 2]
                drot_w = np.clip(0.25 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            elif dep >= 0.0012 and lat <= 0.0010:
                dz_push = 0.28 * (0.0109 - dep) if ang_rad <= 0.032 else 0.0
                dp_tip = -0.50 * lat_vec + dz_push * S[:, 2]
                drot_w = np.clip(0.55 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            elif lat <= 0.00045 and ang_rad <= 0.022:
                dp_tip = -0.65 * lat_vec + 0.42 * (0.0109 - dep) * S[:, 2]
                drot_w = np.clip(0.55 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            else:
                align_err = max(lat / 0.00045, ang_rad / 0.022)
                tgt_dep = -float(np.clip(0.0015 + 0.0012 * (align_err - 1.0), 0.0015, 0.0045))
                dp_tip = -0.75 * lat_vec + 0.45 * (tgt_dep - dep) * S[:, 2]
                drot_w = np.clip(0.55 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            dp_tcp = np.clip(dp_tip - np.cross(drot_w, r_lever), -scale_xyz_m, scale_xyz_m)
        else:
            # Target dep = 11.6 mm: safely above 10.0 mm pass threshold AND 1.0 mm ABOVE 12.6 mm hard bottom!
            align_err = max(lat / 0.00042, ang_rad / 0.018)
            if dep >= 0.0010 and lat <= 0.00075 and ang_rad <= 0.025:
                dp_tip = -0.65 * lat_vec + 0.55 * (0.0116 - dep) * S[:, 2]
            elif align_err <= 1.0:
                dp_tip = -0.80 * lat_vec + 0.52 * (0.0116 - dep) * S[:, 2]
            else:
                tgt_dep = -float(np.clip(0.0012 + 0.0014 * (align_err - 1.0), 0.0012, 0.0050))
                dp_tip = -0.85 * lat_vec + 0.55 * (tgt_dep - dep) * S[:, 2]
            drot_w = np.clip(0.50 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            dp_tcp = np.clip(dp_tip - np.cross(drot_w, r_lever), -scale_xyz_m, scale_xyz_m)
    else:
        delta = task.initial_tip - tip
        hxy = float(np.linalg.norm(delta[:2]))
        dz = float(tip[2] - task.initial_tip[2])
        rotvec_err = Rotation.from_matrix(task.initial_rotation @ Rp.T).as_rotvec()
        ang = float(np.linalg.norm(rotvec_err))
        if smooth_p7:
            # Smooth continuous descent gate (eliminates hxy=0.5mm cliff & Y limit-cycle oscillation)
            if dz <= 0.0022 and hxy <= 0.0018:
                dp_tip = np.array([0.30 * delta[0], 0.30 * delta[1], 0.25 * delta[2]], dtype=np.float64)
                drot_w = np.clip(0.25 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            elif dz > 0.012:
                err_xy = max(0.0, (hxy - 0.00045) / 0.00105)
                err_ang = max(0.0, (ang - 0.020) / 0.025)
                gate = float(np.clip(1.0 - max(err_xy, err_ang), 0.0, 1.0))
                tgt_dz = max(0.016, min(dz, 0.024))
                dz_hold = 0.40 * (tgt_dz - dz)
                dz_desc = float(np.clip(0.45 * delta[2], -0.00115, 0.0))
                dz_cmd = gate * dz_desc + (1.0 - gate) * dz_hold
                dp_tip = np.array([0.58 * delta[0], 0.58 * delta[1], dz_cmd], dtype=np.float64)
                drot_w = np.clip(0.50 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            elif hxy > 0.00095 or ang > 0.032:
                dp_tip = np.array([0.62 * delta[0], 0.62 * delta[1], 0.0014], dtype=np.float64)
                drot_w = np.clip(0.50 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            else:
                dz_cmd = float(np.clip(0.45 * delta[2], -0.0011, 0.0011))
                dp_tip = np.array([0.55 * delta[0], 0.55 * delta[1], dz_cmd], dtype=np.float64)
                drot_w = np.clip(0.45 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            dp_tcp = np.clip(dp_tip - np.cross(drot_w, r_lever), -scale_xyz_m, scale_xyz_m)
        elif soft_p7:
            # Align strictly above narrow 4.80mm holder rim, descend gently, then brake smoothly to zero at floor
            if dz <= 0.0022 and hxy <= 0.0018:
                dp_tip = np.array([0.35 * delta[0], 0.35 * delta[1], 0.25 * delta[2]], dtype=np.float64)
                drot_w = np.clip(0.25 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            elif (hxy > 0.00050 or ang > 0.022) and dz > 0.011:
                tgt_dz = max(0.016, min(dz, 0.024))
                dp_tip = np.array([0.85 * delta[0], 0.85 * delta[1], 0.45 * (tgt_dz - dz)], dtype=np.float64)
                drot_w = np.clip(0.55 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            elif hxy > 0.00090 and dz <= 0.014:
                dp_tip = np.array([0.85 * delta[0], 0.85 * delta[1], 0.0015], dtype=np.float64)
                drot_w = np.clip(0.55 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            else:
                dz_cmd = float(np.clip(0.45 * delta[2], -0.0012, 0.0012))
                dp_tip = np.array([0.75 * delta[0], 0.75 * delta[1], dz_cmd], dtype=np.float64)
                drot_w = np.clip(0.50 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            dp_tcp = np.clip(dp_tip - np.cross(drot_w, r_lever), -scale_xyz_m, scale_xyz_m)
        else:
            if hxy > 0.00045 and tip[2] < task.initial_tip[2] + 0.016:
                dp_tip = np.array([0.80 * delta[0], 0.80 * delta[1], max(0.55 * delta[2], 0.0020)], dtype=np.float64)
            else:
                dp_tip = 0.75 * delta
            drot_w = np.clip(0.50 * rotvec_err, -scale_rot_rad, scale_rot_rad)
            dp_tcp = np.clip(dp_tip - np.cross(drot_w, r_lever), -scale_xyz_m, scale_xyz_m)

    act6 = np.concatenate([
        np.clip(dp_tcp / max(scale_xyz_m, 1e-6), -1.0, 1.0),
        np.clip(drot_w / max(scale_rot_rad, 1e-6), -1.0, 1.0),
    ])
    return act6.astype(np.float32)


def apply_train_only_initial_perturbation(
    env,
    task,
    subtask: str,
    rng: np.random.Generator,
    *,
    soft_p7: bool = False,
    mild_jitter: bool = False,
    p7_q2lift: bool = False,
    p7_inhand_rand: bool = False,
) -> bool:
    """Apply a physically valid pure-friction random handoff perturbation on train episodes ONLY."""
    if p7_inhand_rand and (not mild_jitter) and (rng.random() < 0.85):
        # Train-only in-hand plug slip + tilt in TCP jaw plane (decouples plug pose from gripper TCP for P4 & P7!)
        free_j = [i for i in range(task.m.njnt) if task.m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        if free_j:
            jnt_id = free_j[0]
            qadr = int(task.m.jnt_qposadr[jnt_id])
            vadr = int(task.m.jnt_dofadr[jnt_id])
            task.robot_forward()
            tR = task.rd.site_xmat[task.rtool].reshape(3, 3).copy()
            tp = task.rd.site_xpos[task.rtool].copy()
            if subtask == "p4_insert":
                dy_l = float(rng.uniform(-0.0040, 0.0068))
                dz_l = float(rng.uniform(-0.0038, 0.0020))
                drx = float(np.deg2rad(rng.uniform(-4.5, 4.5)))
                dry = float(np.deg2rad(rng.uniform(-2.5, 2.5)))
            else:
                dy_l = float(rng.uniform(-0.0045, 0.0055))
                dz_l = float(rng.uniform(-0.0035, 0.0020))
                drx = float(np.deg2rad(rng.uniform(-6.5, 6.5)))
                dry = float(np.deg2rad(rng.uniform(-2.5, 2.5)))
            dpos_w = tR @ np.array([0.0, dy_l, dz_l], dtype=np.float64)
            dR_w = tR @ Rotation.from_rotvec(np.array([drx, dry, 0.0], dtype=np.float64)).as_matrix() @ tR.T
            p_cur = task.d.qpos[qadr : qadr + 3].copy()
            q_wxyz = task.d.qpos[qadr + 3 : qadr + 7].copy()
            R_cur = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
            p_new = tp + dR_w @ (p_cur - tp) + dpos_w
            R_new = dR_w @ R_cur
            q_xyzw = Rotation.from_matrix(R_new).as_quat()
            task.d.qpos[qadr : qadr + 3] = p_new
            task.d.qpos[qadr + 3 : qadr + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            task.d.qvel[vadr : vadr + 6] = 0.0
            mujoco.mj_forward(task.m, task.d)
            obs_s = env.observe()
            q_hold = obs_s["observation.state"][6:13].astype(np.float32).copy()
            for _ in range(6):
                cmd_h = np.zeros(8, dtype=np.float32)
                cmd_h[:7] = q_hold
                cmd_h[7] = 0.98
                res_h = env.step(cmd_h)
                env.observe()
                if res_h["terminated"] or res_h["truncated"]:
                    return False
            if float(np.sum(task.d.qpos[task.fq] - task.finger_lower)) < 0.00105:
                return False

    task.robot_forward()
    q_tgt = task.rd.qpos[rcu.ARM_DOF_SLICE].copy()
    if mild_jitter:
        # Gentle live-arrival jitter (+/-2.5mm XY, +/-2mm Z, +/-2deg rot) around real upstream handoffs
        n_steps = int(rng.integers(4, 8))
        dx_step = np.array([
            rng.uniform(-0.00045, 0.00045),
            rng.uniform(-0.00045, 0.00045),
            rng.uniform(-0.00020, 0.00040),
            rng.uniform(-0.0030, 0.0030),
            rng.uniform(-0.0030, 0.0030),
            rng.uniform(-0.0060, 0.0060),
        ], dtype=np.float64)
    elif subtask == "p4_insert":
        # 85% match real upstream Home->P4 handoff distribution; 15% higher-Z P3 hover handoffs
        if rng.random() < 0.15:
            n_steps = int(rng.integers(16, 26))
            dx_step = np.array([
                rng.uniform(-0.00070, 0.00070),
                rng.uniform(-0.00070, 0.00070),
                rng.uniform(0.00100, 0.00180),
                rng.uniform(-0.003, 0.003),
                rng.uniform(-0.003, 0.003),
                rng.uniform(-0.008, 0.008),
            ], dtype=np.float64)
        else:
            n_steps = int(rng.integers(6, 16))
            dx_step = np.array([
                rng.uniform(-0.00075, 0.00075),
                rng.uniform(-0.00075, 0.00075),
                rng.uniform(0.00010, 0.00105),
                rng.uniform(-0.004, 0.004),
                rng.uniform(-0.004, 0.004),
                rng.uniform(-0.011, 0.011),
            ], dtype=np.float64)
    elif p7_q2lift and subtask == "p7_holder":
        # Match real P56-v2 q_p2_lift -> P7 envelope; when in-hand tilt is already randomized, keep arm roll/pitch milder
        n_steps = int(rng.integers(5, 11))
        rp_arm = 0.0045 if p7_inhand_rand else 0.0095
        dx_step = np.array([
            rng.uniform(-0.00040, 0.00040),
            rng.uniform(-0.00040, 0.00040),
            rng.uniform(-0.00025, 0.00045),
            rng.uniform(-rp_arm, rp_arm),
            rng.uniform(-rp_arm, rp_arm),
            rng.uniform(-0.0110, 0.0110),
        ], dtype=np.float64)
    elif soft_p7:
        # Match real Stage 1B P6->P7 handoff envelope (+/-7mm XY, +/-4.5deg tilt/yaw)
        n_steps = int(rng.integers(6, 13))
        dx_step = np.array([
            rng.uniform(-0.00080, 0.00080),
            rng.uniform(-0.00090, 0.00090),
            rng.uniform(-0.00025, 0.00050),
            rng.uniform(-0.0045, 0.0045),
            rng.uniform(-0.0045, 0.0045),
            rng.uniform(-0.010, 0.010),
        ], dtype=np.float64)
    else:
        n_steps = int(rng.integers(4, 10))
        dx_step = np.array([
            rng.uniform(-0.0006, 0.0006),
            rng.uniform(-0.0006, 0.0006),
            rng.uniform(-0.0002, 0.0005),
            rng.uniform(-0.003, 0.003),
            rng.uniform(-0.003, 0.003),
            rng.uniform(-0.008, 0.008),
        ], dtype=np.float64)

    for _ in range(n_steps):
        J = rcu.compute_robot_tcp_jacobian(task)
        dq = np.clip(J.T @ np.linalg.solve(J @ J.T + (0.02 ** 2) * np.eye(6), dx_step), -0.020, 0.020)
        task.robot_forward()
        q_cur = task.rd.qpos[rcu.ARM_DOF_SLICE].copy()
        q_tgt = q_cur + np.clip((q_tgt + dq) - q_cur, -0.024, 0.024)
        cmd = np.zeros(8, dtype=np.float32)
        cmd[:7] = q_tgt
        cmd[7] = 0.98
        res = env.step(cmd)
        env.observe()
        if res["terminated"] or res["truncated"]:
            return False
    fq_sum = float(np.sum(task.d.qpos[task.fq] - task.finger_lower))
    return fq_sum >= 0.00105


def rollout_subtask_episode(
    scene_row: dict,
    snap_path: Path,
    q_home0: np.ndarray,
    provider: Stage1RLTFeatureProvider,
    actor: CartesianResidualActor,
    *,
    subtask: str,
    phase_idx: int,
    max_steps: int,
    explore_sigma: float,
    deterministic: bool,
    allow_expert_takeover: bool,
    trigger_step_range: tuple[int, int],
    apply_perturb: bool,
    rng: np.random.Generator,
    scale_xyz_m: float,
    scale_rot_rad: float,
    q_clamp_rad: float,
    vla_stride: int,
    flow_m: int,
    tmp_dir: Path,
    hires_wrist: bool = True,
    hires_front: bool = False,
    hires_left: bool = False,
    exact_nullspace: bool = False,
    noback_p4: bool = False,
    soft_p7: bool = False,
    smooth_p7: bool = False,
    mild_jitter: bool = False,
    p7_q2lift: bool = False,
    p7_inhand_rand: bool = False,
    p4_extra_hold_steps: int = 0,
) -> tuple[list[dict], dict]:
    seed = int(scene_row["seed"])
    run_dir = tmp_dir / f"ep_{seed}_{int(rng.integers(0, 1_000_000))}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = PolicyExecutor(Path(scene_row["scene"]), run_dir, seed, brightness=float(scene_row["brightness"]))
    device = next(actor.parameters()).device
    actor.eval()
    target_event = "insert" if subtask == "p4_insert" else "return"
    trs: list[dict] = []
    try:
        env.reset(snapshot=snap_path)
        task = env._task
        obs = env.observe()
        if apply_perturb and ((not deterministic) or mild_jitter):
            ok_p = apply_train_only_initial_perturbation(
                env, task, subtask, rng,
                soft_p7=soft_p7, mild_jitter=mild_jitter, p7_q2lift=p7_q2lift, p7_inhand_rand=p7_inhand_rand,
            )
            if not ok_p:
                env.reset(snapshot=snap_path)
            obs = env.observe()

        raw14 = obs["observation.state"][:14].astype(np.float32)
        tcp_entry = raw14[:3].copy()
        q_tgt = raw14[6:13].astype(np.float64).copy()
        last_cmd_q7 = raw14[6:13].copy()
        q_hist = [raw14[6:13].copy()] * 5
        prev_act6 = np.zeros(6, dtype=np.float32)
        stall_steps = 0
        pot_prev, _ = compute_subtask_potential(task, subtask)

        cached_z = cached_ref_rel32 = None
        steps_since_vla = vla_stride
        success = False
        safety_stop = False
        step_done = 0
        p7_hold_steps = 0
        p4_post_insert_steps = 0
        use_geom_return = bool(
            p7_q2lift and subtask == "p7_holder" and (not deterministic)
            and ("withdraw" not in env.outcomes()["physical"].get("event_times", {}))
        )
        expert_steps = 0
        takeover_active = False
        ep_allow_takeover = bool(allow_expert_takeover and (not deterministic))
        tk_trigger_step = int(rng.integers(trigger_step_range[0], max(trigger_step_range[0] + 1, trigger_step_range[1])))
        best_metric = dict(min_lat=999.0, max_dep=-999.0, min_hdz=999.0, min_hxy=999.0)
        ep_drift = rng.normal(0.0, explore_sigma * 0.35, size=6).astype(np.float32) if not deterministic else np.zeros(6, dtype=np.float32)

        for step_i in range(0, max_steps, 2):
            if cached_z is None or steps_since_vla >= vla_stride:
                cached_z, _, cached_ref_rel32 = provider.infer(
                    obs, q_home0, phase_idx, step_seed=7317 + step_i, flow_m=flow_m
                )
                steps_since_vla = 0

            w_crop = extract_wrist_128(task, obs, hires_wrist=hires_wrist)
            f_crop = extract_front_96(task, obs, subtask, hires_front=hires_front)
            l_crop = extract_left_96(task, obs, subtask, hires_left=hires_left)
            p_feat = rcu.extract_contact_proprio(
                task, obs["observation.state"], last_cmd_q7, q_hist[0], tcp_entry, stall_steps, prev_act6
            )
            g_feat = compute_train_only_geom6(task, subtask)

            # Strictly TRAIN-ONLY stalled_progress / HG-DAgger expert takeover!
            if ep_allow_takeover and (not takeover_active) and (stall_steps >= 8 or step_i >= tk_trigger_step):
                takeover_active = True

            is_expert_step = False
            if takeover_active:
                act6 = compute_train_only_expert_act6(
                    task, subtask, scale_xyz_m=scale_xyz_m, scale_rot_rad=scale_rot_rad,
                    compliant_p4=exact_nullspace, noback_p4=noback_p4, soft_p7=soft_p7, smooth_p7=smooth_p7,
                )
                is_expert_step = True
                expert_steps += 1
            else:
                with torch.no_grad():
                    l_t = torch.from_numpy(l_crop[None]).to(device) if l_crop is not None else None
                    a_mean = actor(
                        torch.from_numpy(cached_z[None]).to(device),
                        torch.from_numpy(p_feat[None]).to(device),
                        torch.from_numpy(cached_ref_rel32[None]).to(device),
                        torch.from_numpy(w_crop[None]).to(device),
                        torch.from_numpy(f_crop[None]).to(device),
                        l_t,
                    )[0].cpu().numpy().astype(np.float32)
                if deterministic:
                    act6 = a_mean
                else:
                    sig = explore_sigma * (1.30 if stall_steps >= 8 else 1.0)
                    act6 = np.clip(a_mean + ep_drift + rng.normal(0.0, sig, size=6).astype(np.float32), -1.0, 1.0)

            dq = step_persistent_qtgt_dq(
                task, act6, scale_xyz_m=scale_xyz_m, scale_rot_rad=scale_rot_rad, max_dq_rad=0.025, exact_nullspace=exact_nullspace
            )
            terminated_now = False
            for _ in range(2):
                q_cur = obs["observation.state"][6:13].astype(np.float64)
                q_tgt = q_tgt + 0.5 * dq
                q_tgt = q_cur + np.clip(q_tgt - q_cur, -q_clamp_rad, q_clamp_rad)
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
                    elif subtask == "p4_insert" and deterministic and p4_extra_hold_steps > 0:
                        fq_s = float(np.sum(task.d.qpos[task.fq] - task.finger_lower))
                        if ("withdraw" in phys.get("event_times", {})) or fq_s < 0.00105:
                            success = False
                            terminated_now = True
                            break
                        p4_post_insert_steps += 1
                        if p4_post_insert_steps >= p4_extra_hold_steps:
                            success = True
                            terminated_now = True
                            break
                    else:
                        success = True
                        terminated_now = True
                        break
                if res["terminated"] or res["truncated"]:
                    safety_stop = (phys.get("invalid_reason") == "safety_stop")
                    terminated_now = True
                    break

            pot_next, info_next = compute_subtask_potential(task, subtask)
            if subtask == "p4_insert":
                best_metric["min_lat"] = min(best_metric["min_lat"], info_next["lat_mm"])
                best_metric["max_dep"] = max(best_metric["max_dep"], info_next["dep_mm"])
            else:
                best_metric["min_hxy"] = min(best_metric["min_hxy"], info_next["hxy_mm"])
                best_metric["min_hdz"] = min(best_metric["min_hdz"], info_next["hdz_mm"])

            r_step = float(pot_next - pot_prev) - 0.005 * float(np.mean(act6 ** 2)) - (0.04 if stall_steps >= 10 else 0.0)
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
            prev_z, prev_ref = cached_z, cached_ref_rel32
            if not terminated_now and steps_since_vla >= vla_stride:
                cached_z, _, cached_ref_rel32 = provider.infer(
                    obs, q_home0, phase_idx, step_seed=7317 + step_i + 2, flow_m=flow_m
                )
                steps_since_vla = 0

            next_w = extract_wrist_128(task, obs, hires_wrist=hires_wrist)
            next_f = extract_front_96(task, obs, subtask, hires_front=hires_front)
            next_l = extract_left_96(task, obs, subtask, hires_left=hires_left)
            next_p = rcu.extract_contact_proprio(
                task, obs["observation.state"], last_cmd_q7, q_hist[0], tcp_entry, stall_steps, act6
            )
            next_g = compute_train_only_geom6(task, subtask)
            tr_rec = dict(
                z=prev_z, p=p_feat, ref=prev_ref, w=w_crop, f=f_crop, g=g_feat, a=act6.astype(np.float32),
                r=float(r_step), nz=cached_z, np=next_p, nref=cached_ref_rel32, nw=next_w, nf=next_f, ng=next_g,
                done=float(done_flag), ep_succ=False, is_expert=bool(is_expert_step), mc=0.0,
            )
            if hires_left:
                tr_rec["l"] = l_crop
                tr_rec["nl"] = next_l
            trs.append(tr_rec)
            prev_act6 = act6.copy()
            pot_prev = pot_next
            if terminated_now:
                break

        G = 0.0
        for tr in reversed(trs):
            G = tr["r"] + 0.98 * G * (1.0 - tr["done"])
            tr["mc"] = float(G)
            tr["ep_succ"] = bool(success)

        rep = dict(
            seed=seed, subtask=subtask, success=bool(success), safety_stop=bool(safety_stop),
            steps=step_done, expert_steps=expert_steps, ep_return=round(sum(x["r"] for x in trs), 3),
            **{k: round(v, 3) for k, v in best_metric.items() if abs(v) < 500},
        )
        return trs, rep
    finally:
        close_hires_wrist_renderer(env._task)
        env.close()
        shutil.rmtree(run_dir, ignore_errors=True)


def build_all_train_perturbed_pool(subtask: str, scenes_root: Path, val_seeds: set[int], *, p7_q2lift: bool = False) -> list[dict]:
    """Load/cache all train scenes for the exact scenes_root (up to 1,049 v3_fresh scenes, strictly disjoint from val/test)."""
    sr = Path(scenes_root)
    cache_dir = H2 / "handoff_pools"
    cache_dir.mkdir(parents=True, exist_ok=True)
    v3_1200_manifest = Path("/mlplatform/jy/v3_fresh_nvme/raw_full_1200/manifest.json")
    v3_1200_scenes = Path("/mlplatform/jy/v3_fresh_nvme/train_scenes_v3_1200.json")
    tag = "v3fresh1200_" if ("v3" in sr.name and v3_1200_manifest.exists() and v3_1200_scenes.exists()) else ""
    q2_sfx = "_q2lift" if (p7_q2lift and subtask == "p7_holder") else ""
    cache_path = cache_dir / f"all_train_{tag}{sr.name}_{subtask}{q2_sfx}.json"
    if cache_path.exists():
        items = json.loads(cache_path.read_text())
        return [x for x in items if int(x["scene"]["seed"]) not in val_seeds]
    if subtask == "p4_insert":
        snap_fname = "cycle_phase_align_at_mouth.npz"
    elif p7_q2lift:
        snap_fname = "cycle_phase_translate_above_port.npz"
    else:
        snap_fname = "cycle_phase_replace_in_holder.npz"
    items = []
    cand_manifests = [
        v3_1200_manifest,
        Path("/mlplatform/jy/v3_d8_nvme/raw_v3_d8_full372/manifest.json"),
        H2 / "datasets/raw_v3_d8_full372/manifest.json",
    ]
    raw_v3_manifest = next((m for m in cand_manifests if m.exists()), cand_manifests[0])
    test_seeds = set()
    test_json = sr / "stage2_test_scenes_v3.json"
    if test_json.exists():
        test_seeds = {int(s["seed"]) for s in json.loads(test_json.read_text())}
    forbidden_seeds = set(val_seeds) | test_seeds
    if "v3" in sr.name and raw_v3_manifest.exists():
        tr_json = v3_1200_scenes if (raw_v3_manifest == v3_1200_manifest and v3_1200_scenes.exists()) else (sr / "stage2_train_scenes_v3.json")
        by_seed = {int(s["seed"]): s for s in json.loads(tr_json.read_text())}
        for rec in sorted(json.loads(raw_v3_manifest.read_text()), key=lambda r: int(r["seed"])):
            if not rec.get("ok"):
                continue
            seed = int(rec["seed"])
            if seed in forbidden_seeds or seed not in by_seed:
                continue
            rec_path = Path(rec["path"])
            snap = rec_path / snap_fname
            demo = rec_path / "demonstration_full.npz"
            if snap.exists() and demo.exists():
                with np.load(demo, allow_pickle=True) as d:
                    qh = d["observation.state"][0, 6:13].astype(np.float32).tolist()
                items.append(dict(scene=dict(by_seed[seed]), snapshot=str(snap), q_home0=qh, source="all_v3_train_perturb"))
    else:
        from eval_v2_fast import select_v2_rows
        rows = select_v2_rows("train", None, 999, sr)
        for row in rows:
            sc = row["scene"]
            seed = int(sc["seed"])
            if seed in val_seeds:
                continue
            rec_path = Path(row["record"]["path"])
            snap = rec_path / snap_fname
            demo = rec_path / "demonstration_full.npz"
            if snap.exists() and demo.exists():
                with np.load(demo, allow_pickle=True) as d:
                    qh = d["observation.state"][0, 6:13].astype(np.float32).tolist()
                items.append(dict(scene=sc, snapshot=str(snap), q_home0=qh, source="all372_perturb"))
    cache_path.write_text(json.dumps(items, indent=2) + "\n")
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", choices=("p4_insert", "p7_holder"), required=True)
    ap.add_argument("--stage1-config-dir", default=str(EXP12 / "configs"))
    ap.add_argument("--stage1-config-name", default="phase12_sft8")
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--rlt-adapter-pt", default="")
    ap.add_argument("--train-pool-json", required=True)
    ap.add_argument("--val-pool-json", required=True)
    ap.add_argument("--scenes-root", default=str(H2 / "scenes_v2_keepvis"))
    ap.add_argument("--use-all-train-scenes", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rounds", type=int, default=36)
    ap.add_argument("--eps-per-round", type=int, default=16)
    ap.add_argument("--updates-per-round", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--flow-m", type=int, default=1)
    ap.add_argument("--vla-stride", type=int, default=6)
    ap.add_argument("--scale-xyz-mm", type=float, default=1.8)
    ap.add_argument("--scale-rot-rad", type=float, default=0.022)
    ap.add_argument("--q-clamp-rad", type=float, default=0.028)
    ap.add_argument("--init-sigma", type=float, default=0.22)
    ap.add_argument("--expert-takeover-prob", type=float, default=0.70)
    ap.add_argument("--preload-buffer-pt", default="")
    ap.add_argument("--pretrain-steps", type=int, default=4500)
    ap.add_argument("--q-weight-max", type=float, default=0.06)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-fast-limit", type=int, default=12)
    ap.add_argument("--hires-wrist-roi", action="store_true", default=False)
    ap.add_argument("--hires-front-roi", action="store_true", default=False)
    ap.add_argument("--hires-left-roi", action="store_true", default=False)
    ap.add_argument("--exact-nullspace", action="store_true", default=False)
    ap.add_argument("--noback-p4", action="store_true", default=False)
    ap.add_argument("--soft-p7", action="store_true", default=False)
    ap.add_argument("--smooth-p7", action="store_true", default=False)
    ap.add_argument("--p7-q2lift", action="store_true", default=False)
    ap.add_argument("--p7-inhand-rand", "--inhand-rand", dest="p7_inhand_rand", action="store_true", default=False)
    ap.add_argument("--no-shift-aug", action="store_true", default=False)
    ap.add_argument("--aux-geom-weight", type=float, default=0.6)
    ap.add_argument("--aux-xyrot-boost", action="store_true", default=False)
    ap.add_argument("--ref-dropout-prob", type=float, default=0.20)
    ap.add_argument("--proprio-dropout-prob", type=float, default=0.0)
    ap.add_argument("--prev-act-drop-prob", type=float, default=0.0)
    ap.add_argument("--real-tr-prob", type=float, default=0.35)
    ap.add_argument("--real-jitter-prob", type=float, default=0.0)
    ap.add_argument("--eval-jitter-val", action="store_true", default=False)
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--lr-hi", type=float, default=0.0)
    ap.add_argument("--p4-extra-hold-steps", type=int, default=0)
    ap.add_argument("--ema-eval", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("/dev/shm/jy_h2_tmp") / f"rlt_{args.subtask}_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    train_pool = json.loads(Path(args.train_pool_json).read_text())
    val_pool_full = json.loads(Path(args.val_pool_json).read_text())
    tr_seeds = {int(x["scene"]["seed"]) for x in train_pool}
    va_seeds = {int(x["scene"]["seed"]) for x in val_pool_full}
    sr_path = Path(args.scenes_root)
    test_j = sr_path / ("stage2_test_scenes_v3.json" if (sr_path / "stage2_test_scenes_v3.json").exists() else "stage2_test_scenes_v2.json")
    te_seeds = {int(s["seed"]) for s in json.loads(test_j.read_text())} if test_j.exists() else set()
    v100_j = sr_path / "val100_unseen_scenes_v3.json"
    v100_seeds = {int(s["seed"]) for s in json.loads(v100_j.read_text())} if v100_j.exists() else set()
    v34_j = sr_path / "stage2_validation_scenes_v3.json"
    v34_seeds = {int(s["seed"]) for s in json.loads(v34_j.read_text())} if v34_j.exists() else set()
    forbid_seeds = va_seeds | v34_seeds | v100_seeds | te_seeds
    assert tr_seeds.isdisjoint(forbid_seeds), "TRAIN/VAL/TEST SEED LEAKAGE DETECTED!"

    real_tr = [x for x in train_pool if x.get("source") == "real_home_handoff"]
    all372_tr = build_all_train_perturbed_pool(args.subtask, sr_path, forbid_seeds, p7_q2lift=bool(args.p7_q2lift)) if args.use_all_train_scenes else []
    if all372_tr:
        a_seeds = {int(x["scene"]["seed"]) for x in all372_tr}
        assert a_seeds.isdisjoint(va_seeds), "ALL372/VAL LEAKAGE DETECTED!"
    print(f"[INIT] subtask={args.subtask} real_tr={len(real_tr)} all372_tr={len(all372_tr)} val_full={len(val_pool_full)} vla_stride={args.vla_stride} flow_m={args.flow_m}", flush=True)

    val_fast = [val_pool_full[i] for i in np.linspace(0, len(val_pool_full) - 1, min(args.val_fast_limit, len(val_pool_full)), dtype=int)]
    phase_idx = 4 if args.subtask == "p4_insert" else 7
    provider = Stage1RLTFeatureProvider(
        Path(args.stage1_config_dir), args.stage1_config_name, args.stage1_ckpt, device, rlt_adapter_pt=args.rlt_adapter_pt
    )
    trainer = ResidualRLPDTrainer(
        device,
        q_weight_max=args.q_weight_max,
        aux_geom_weight=args.aux_geom_weight,
        aux_xyrot_boost=bool(args.aux_xyrot_boost),
        ref_dropout_prob=args.ref_dropout_prob,
        proprio_dropout_prob=args.proprio_dropout_prob,
        prev_act_drop_prob=args.prev_act_drop_prob,
        shift_aug=(not args.no_shift_aug),
        use_left_wrist=bool(args.hires_left_roi),
    )
    if args.init_ckpt:
        init_ck = torch.load(str(args.init_ckpt), map_location="cpu", weights_only=False)
        trainer.actor.load_state_dict(init_ck["actor"], strict=True)
        trainer.target_actor.load_state_dict(init_ck["actor"], strict=True)
        if "critic" in init_ck:
            trainer.critic.load_state_dict(init_ck["critic"], strict=True)
            trainer.target_critic.load_state_dict(init_ck["critic"], strict=True)
        print(f"[INIT_CKPT] Warm-started actor+critic from {args.init_ckpt} (prev val_rate={init_ck.get('val_rate')})", flush=True)
    demo_buf = CompactDemoBuffer(args.preload_buffer_pt) if args.preload_buffer_pt else None
    if demo_buf is not None:
        buf_meta = torch.load(str(args.preload_buffer_pt), map_location="cpu", weights_only=False)
        buf_hires = bool(buf_meta.get("hires_wrist_roi", False))
        buf_hfront = bool(buf_meta.get("hires_front_roi", False))
        buf_hleft = bool(buf_meta.get("hires_left_roi", False))
        del buf_meta
        assert buf_hires == bool(args.hires_wrist_roi), (
            f"Demo buffer hires_wrist_roi={buf_hires} != --hires-wrist-roi={args.hires_wrist_roi}!"
        )
        assert buf_hfront == bool(args.hires_front_roi), (
            f"Demo buffer hires_front_roi={buf_hfront} != --hires-front-roi={args.hires_front_roi}!"
        )
        assert buf_hleft == bool(args.hires_left_roi), (
            f"Demo buffer hires_left_roi={buf_hleft} != --hires-left-roi={args.hires_left_roi}!"
        )
        print(f"[PRELOAD] Loaded {demo_buf.size} demo transitions ({len(demo_buf.succ_indices)} succ, hires_wrist={buf_hires}, hires_front={buf_hfront}, hires_left={buf_hleft}) from {args.preload_buffer_pt}", flush=True)

    all_buf: list[dict] = []
    succ_buf: list[dict] = []
    max_buf = 36000
    best_val_score = (-1.0, -999, -999.0)
    history = []

    def eval_val(
        tag: str, pool: list[dict], eval_actor: CartesianResidualActor | None = None, *, eval_jitter: bool = False
    ) -> tuple[float, dict]:
        act_use = trainer.actor if eval_actor is None else eval_actor
        eval_rng = np.random.default_rng(20261125)
        reps = []
        for item in pool:
            _, r = rollout_subtask_episode(
                item["scene"], Path(item["snapshot"]), np.asarray(item["q_home0"], dtype=np.float32),
                provider, act_use,
                subtask=args.subtask, phase_idx=phase_idx, max_steps=args.max_steps,
                explore_sigma=0.0, deterministic=True, allow_expert_takeover=False,
                trigger_step_range=(9999, 10000), apply_perturb=eval_jitter, rng=eval_rng,
                scale_xyz_m=args.scale_xyz_mm * 1e-3, scale_rot_rad=args.scale_rot_rad,
                q_clamp_rad=args.q_clamp_rad, vla_stride=args.vla_stride, flow_m=args.flow_m, tmp_dir=tmp_dir,
                hires_wrist=bool(args.hires_wrist_roi),
                hires_front=bool(args.hires_front_roi),
                hires_left=bool(args.hires_left_roi),
                exact_nullspace=bool(args.exact_nullspace),
                noback_p4=bool(args.noback_p4),
                soft_p7=bool(args.soft_p7),
                mild_jitter=eval_jitter,
                p4_extra_hold_steps=int(args.p4_extra_hold_steps),
            )
            assert r["expert_steps"] == 0, "Eval must NEVER invoke expert takeover!"
            reps.append(r)
        n_ok = sum(1 for r in reps if r["success"])
        n_saf = sum(1 for r in reps if r["safety_stop"])
        rate = n_ok / max(1, len(reps))
        avg_prog = float(np.mean([r.get("max_dep", -r.get("min_hdz", 30.0)) for r in reps]))
        summary = dict(tag=tag, success=n_ok, total=len(reps), rate=round(rate, 4), safety_stops=n_saf, avg_prog=round(avg_prog, 3), reports=reps)
        (out_dir / f"val_{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[VAL {tag}] {args.subtask}: {n_ok}/{len(reps)} ({rate*100:.1f}%) safety={n_saf} avg_prog={avg_prog:.2f}", flush=True)
        return rate, summary

    arch_ver = "v4_tristereo640_qtgt" if args.hires_left_roi else ("v3_hires_wrist_qtgt" if args.hires_wrist_roi else "v2_stereo_qtgt")
    t0 = time.time()
    if demo_buf is not None and args.pretrain_steps > 0:
        print(f"[PRETRAIN] Running {args.pretrain_steps} offline RLPD+AuxGeom+BC steps (q_w=0, aux_w={args.aux_geom_weight})...", flush=True)
        for p_blk in range(0, args.pretrain_steps, 500):
            n_st = min(500, args.pretrain_steps - p_blk)
            pm = trainer.update(all_buf, succ_buf, rng, steps=n_st, batch_size=args.batch_size, demo_buf=demo_buf, allow_q_grad=False)
            print(f"  [PRETRAIN step {p_blk + n_st}/{args.pretrain_steps}] elapsed={time.time()-t0:.1f}s {json.dumps(pm)}", flush=True)
        val_rate, val_sum = eval_val("round00_pretrain_full", val_pool_full)
        eff_rate, eff_saf, eff_prog = val_rate, val_sum["safety_stops"], val_sum["avg_prog"]
        if args.eval_jitter_val and val_rate >= 0.80:
            j_rate, j_sum = eval_val("round00_pretrain_jitter_full", val_pool_full, eval_jitter=True)
            eff_rate = 0.5 * (val_rate + j_rate)
            eff_saf = val_sum["safety_stops"] + j_sum["safety_stops"]
            eff_prog = 0.5 * (val_sum["avg_prog"] + j_sum["avg_prog"])
        best_val_score = (eff_rate, -eff_saf, eff_prog)
        torch.save(
            dict(
                arch_version=arch_ver,
                actor=trainer.actor.state_dict(),
                critic=trainer.critic.state_dict(),
                args=vars(args),
                val_rate=val_rate,
                eff_rate=eff_rate,
                round=0,
            ),
            out_dir / "best_rlt_stage2.pt",
        )
        # Drop actor LR gently for online HG-DAgger + RLPD fine-tuning
        for pg in trainer.actor_opt.param_groups:
            pg["lr"] = 1.5e-4
    elif args.init_ckpt:
        val_rate, val_sum = eval_val("round00_init_full", val_pool_full)
        eff_rate, eff_saf, eff_prog = val_rate, val_sum["safety_stops"], val_sum["avg_prog"]
        if args.eval_jitter_val:
            j_rate, j_sum = eval_val("round00_init_jitter_full", val_pool_full, eval_jitter=True)
            eff_rate = 0.5 * (val_rate + j_rate)
            eff_saf = val_sum["safety_stops"] + j_sum["safety_stops"]
            eff_prog = 0.5 * (val_sum["avg_prog"] + j_sum["avg_prog"])
        best_val_score = (eff_rate, -eff_saf, eff_prog)
        torch.save(
            dict(
                arch_version=arch_ver,
                actor=trainer.actor.state_dict(),
                critic=trainer.critic.state_dict(),
                args=vars(args),
                val_rate=val_rate,
                eff_rate=eff_rate,
                round=0,
            ),
            out_dir / "best_rlt_stage2.pt",
        )
        for pg in trainer.actor_opt.param_groups:
            pg["lr"] = float(args.lr_hi) if args.lr_hi > 0.0 else 1.0e-4

    for rd in range(1, args.rounds + 1):
        if demo_buf is not None and args.ema_eval:
            # Gentle cosine LR schedule across online rounds to prevent late-round oscillation
            if args.lr_hi > 0.0:
                lr_hi, lr_lo = float(args.lr_hi), float(args.lr_hi) * 0.30
            else:
                lr_hi, lr_lo = (1.0e-4, 3e-5) if args.init_ckpt else (1.5e-4, 5e-5)
            frac = float(rd - 1) / max(1.0, float(args.rounds - 1))
            lr_now = lr_lo + 0.5 * (lr_hi - lr_lo) * (1.0 + float(np.cos(np.pi * frac)))
            for pg in trainer.actor_opt.param_groups:
                pg["lr"] = lr_now
        sigma = max(0.04, args.init_sigma * (0.92 ** (rd - 1)))
        if demo_buf is not None:
            # Online HG-DAgger + RLPD after pretraining: mostly autonomous, rescue ONLY when truly stalled!
            tk_prob = max(0.25, 0.55 * (0.92 ** (rd - 1)))
            trig_range = (60, 120)
        elif rd == 1:
            tk_prob = 0.85
            trig_range = (0, 12)
        elif rd <= 4:
            tk_prob = max(0.55, args.expert_takeover_prob * (0.94 ** (rd - 1)))
            trig_range = (10, 32)
        else:
            tk_prob = max(0.20, args.expert_takeover_prob * (0.93 ** (rd - 1)))
            trig_range = (18, 56)

        rd_auto_ok = rd_auto_n = rd_exp_ok = rd_exp_n = rd_saf = 0
        for _ in range(args.eps_per_round):
            use_perturb = False
            use_mild_jitter = False
            if args.subtask == "p4_insert":
                if real_tr and all372_tr:
                    if rng.random() < args.real_tr_prob:
                        item = real_tr[int(rng.integers(0, len(real_tr)))]
                        if args.real_jitter_prob > 0.0 and rng.random() < args.real_jitter_prob:
                            use_perturb = True
                            use_mild_jitter = True
                    else:
                        item = all372_tr[int(rng.integers(0, len(all372_tr)))]
                        use_perturb = True
                elif real_tr:
                    item = real_tr[int(rng.integers(0, len(real_tr)))]
                    if args.real_jitter_prob > 0.0 and rng.random() < args.real_jitter_prob:
                        use_perturb = True
                        use_mild_jitter = True
                    if args.real_jitter_prob > 0.0 and rng.random() < args.real_jitter_prob:
                        use_perturb = True
                        use_mild_jitter = True
                else:
                    item = train_pool[int(rng.integers(0, len(train_pool)))]
                    use_perturb = True
            else:
                if real_tr and all372_tr:
                    if rng.random() < args.real_tr_prob:
                        item = real_tr[int(rng.integers(0, len(real_tr)))]
                        use_perturb = bool(rng.random() < max(0.25, args.real_jitter_prob))
                        use_mild_jitter = use_perturb and (args.real_jitter_prob > 0.0)
                    else:
                        item = all372_tr[int(rng.integers(0, len(all372_tr)))]
                        use_perturb = bool(rng.random() < (0.90 if args.p7_inhand_rand else 0.70))
                elif real_tr:
                    item = real_tr[int(rng.integers(0, len(real_tr)))]
                else:
                    pool_src = all372_tr if all372_tr else train_pool
                    item = pool_src[int(rng.integers(0, len(pool_src)))]
                    use_perturb = bool(rng.random() < 0.50)

            use_tk = bool(rng.random() < tk_prob)
            trs, rep = rollout_subtask_episode(
                item["scene"], Path(item["snapshot"]), np.asarray(item["q_home0"], dtype=np.float32),
                provider, trainer.actor,
                subtask=args.subtask, phase_idx=phase_idx, max_steps=args.max_steps,
                explore_sigma=sigma, deterministic=False, allow_expert_takeover=use_tk,
                trigger_step_range=trig_range, apply_perturb=use_perturb, rng=rng,
                scale_xyz_m=args.scale_xyz_mm * 1e-3, scale_rot_rad=args.scale_rot_rad,
                q_clamp_rad=args.q_clamp_rad, vla_stride=args.vla_stride, flow_m=args.flow_m, tmp_dir=tmp_dir,
                hires_wrist=bool(args.hires_wrist_roi),
                hires_front=bool(args.hires_front_roi),
                hires_left=bool(args.hires_left_roi),
                exact_nullspace=bool(args.exact_nullspace),
                noback_p4=bool(args.noback_p4),
                soft_p7=bool(args.soft_p7),
                smooth_p7=bool(args.smooth_p7),
                mild_jitter=use_mild_jitter,
                p7_q2lift=bool(args.p7_q2lift),
                p7_inhand_rand=bool(args.p7_inhand_rand),
            )
            all_buf.extend(trs)
            if rep["success"]:
                succ_buf.extend(trs)
            if rep["expert_steps"] > 0:
                rd_exp_n += 1
                rd_exp_ok += int(rep["success"])
            else:
                rd_auto_n += 1
                rd_auto_ok += int(rep["success"])
            if rep["safety_stop"]:
                rd_saf += 1

        if len(all_buf) > max_buf:
            all_buf = all_buf[-max_buf:]
        if len(succ_buf) > max_buf // 2:
            succ_buf = succ_buf[-(max_buf // 2):]

        upd = trainer.update(all_buf, succ_buf, rng, steps=args.updates_per_round, batch_size=args.batch_size, demo_buf=demo_buf, allow_q_grad=(rd >= 2))
        gate_eval = trainer.eval_critic_gate(all_buf)
        rec = dict(
            round=rd, elapsed_s=round(time.time() - t0, 1), sigma=round(sigma, 3), tk_prob=round(tk_prob, 2),
            auto_ok=f"{rd_auto_ok}/{rd_auto_n}", exp_ok=f"{rd_exp_ok}/{rd_exp_n}", train_saf=rd_saf,
            buf_all=len(all_buf), buf_succ=len(succ_buf), gate_eval=gate_eval, **upd,
        )
        if (demo_buf is not None and rd % 2 == 0) or rd % 4 == 0 or rd == args.rounds:
            val_rate, val_sum = eval_val(f"round{rd:02d}_full", val_pool_full)
            rec["val_full_rate"] = round(val_rate, 4)
            cand_list = [(val_rate, val_sum, trainer.actor, "actor")]
            if args.ema_eval:
                ema_rate, ema_sum = eval_val(f"round{rd:02d}_ema_full", val_pool_full, eval_actor=trainer.target_actor)
                rec["val_ema_rate"] = round(ema_rate, 4)
                cand_list.append((ema_rate, ema_sum, trainer.target_actor, "ema"))
            for c_rate, c_sum, c_act, c_src in cand_list:
                eff_rate = c_rate
                eff_saf = c_sum["safety_stops"]
                eff_prog = c_sum["avg_prog"]
                if args.eval_jitter_val and c_rate >= 0.80:
                    j_rate, j_sum = eval_val(f"round{rd:02d}_{c_src}_jitter_full", val_pool_full, eval_actor=c_act, eval_jitter=True)
                    rec[f"val_{c_src}_jitter_rate"] = round(j_rate, 4)
                    eff_rate = 0.5 * (c_rate + j_rate)
                    eff_saf = c_sum["safety_stops"] + j_sum["safety_stops"]
                    eff_prog = 0.5 * (c_sum["avg_prog"] + j_sum["avg_prog"])
                score = (eff_rate, -eff_saf, eff_prog)
                if score >= best_val_score:
                    best_val_score = score
                    torch.save(
                        dict(
                            arch_version=arch_ver,
                            actor=c_act.state_dict(),
                            critic=trainer.critic.state_dict(),
                            args=vars(args),
                            val_rate=c_rate,
                            eff_rate=eff_rate,
                            val_src=c_src,
                            round=rd,
                        ),
                        out_dir / "best_rlt_stage2.pt",
                    )
        elif rd % 2 == 0:
            val_rate, _ = eval_val(f"round{rd:02d}_fast", val_fast)
            rec["val_fast_rate"] = round(val_rate, 4)

        history.append(rec)
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"[RD {rd:02d}] {json.dumps(rec)}", flush=True)

    best_ck = torch.load(out_dir / "best_rlt_stage2.pt", map_location="cpu")
    trainer.actor.load_state_dict(best_ck["actor"], strict=True)
    eval_val("final_best_full_val", val_pool_full)


if __name__ == "__main__":
    main()
