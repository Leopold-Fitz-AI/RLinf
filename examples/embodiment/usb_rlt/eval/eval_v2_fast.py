# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fast, exact fp32+autocast evaluation on usb_holder_v2 with per-phase flow-M & SafeReversibleSwitcher."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
import av
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from openpi import transforms as T
from torch.distributed import checkpoint as dcp
from torch.utils.data._utils.collate import default_collate

ROOT = Path("/vla-cd/jy/v12b_usb_cycle_v1")
EXP12 = ROOT / "clean_rel256_phase12_v1"
EXP13 = ROOT / "clean_rel256_phase13_v1"
H2 = ROOT / "holder_v2_rlt"

for p in [
    "/root/v12b_usb_cycle_v1/runtime",
    str(H2 / "code"),
    str(EXP13 / "code"),
    str(EXP12 / "code"),
    str(EXP13 / "dependencies"),
    str(EXP13 / "rlinf"),
    str(ROOT),
    str(ROOT / "approach_audit_v1"),
    str(ROOT / "revisions/teacher_axis_return_v1"),
    "/root/v12b_usb_cycle_v1/base_code/scripts",
    "/root/v12b_usb_cycle_v1/base_code/src",
]:
    if p not in sys.path:
        sys.path.append(p)

from policy_executor import PolicyExecutor
from evaluate_closed_loop_phase13 import (
    DEFAULT_PHASE_EXEC_STEPS as P13_EXEC_STEPS,
    NonPrivilegedPhase13Switcher,
    parse_phase_int_map,
)
from evaluate_closed_loop_phase12 import (
    DEFAULT_PHASE_EXEC_STEPS as P12_EXEC_STEPS,
    NonPrivilegedPhase12Switcher,
)
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.modules import model as openpi_model
from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms
import usb_rel256_phase12 as usb12
import usb_rel256_phase13 as usb13

CAMERAS = ("front", "left_wrist", "wrist")
NOISE = 7317


class SafeFingerGuardMixin:
    """Real Dex1-1 finger-encoder empty-grasp guard + safe vertical retract retry (zero privileged pose)."""

    def init_guard(self, max_grasp_retries: int = 2) -> None:
        self.max_grasp_retries = int(max_grasp_retries)
        self.grasp_retries = 0
        self.empty_grasp_events = 0
        self.retract_steps = 0
        self.retract_q_target: np.ndarray | None = None
        self.p1_entry_q: np.ndarray | None = None
        self.p4_entry_z: float | None = None
        self.open_lift_steps = 0

    def guard_step(self, step: int, raw14: np.ndarray, fq_margin: np.ndarray, q_home0: np.ndarray):
        """Returns (override_phase_or_None, override_cmd_or_None, block_p2_advance, block_p4_advance)."""
        grip = float(raw14[13])
        q_cur = np.asarray(raw14[6:13], dtype=np.float32)
        fq_sum = float(np.sum(fq_margin))
        fq_min = float(np.min(fq_margin))

        # Remember clean pregrasp joint posture when entering P1 (directly above holder — safe vertical lift target!)
        if self.phase == 1 and self.p1_entry_q is None:
            self.p1_entry_q = q_cur.copy()

        if self.retract_steps > 0:
            self.retract_steps -= 1
            tgt = self.retract_q_target if self.retract_q_target is not None else q_cur
            cmd = np.zeros(8, dtype=np.float32)
            cmd[:7] = q_cur + np.clip(tgt - q_cur, -0.015, 0.015)
            cmd[7] = 0.0
            if self.retract_steps == 0:
                self.phase = 1
                self.settle_steps = 0
                self.transitions.append(dict(
                    step=int(step), from_phase=0, to_phase=1,
                    reason=f"empty_grasp_retry_{self.grasp_retries}",
                    tcp=[round(float(x), 4) for x in raw14[:3]], grip=0.0,
                ))
            return self.phase, cmd, True, False

        # Empty-grasp detection BEFORE hitting lower - 0.00015 safety guard!
        if self.phase in (1, 2) and grip >= 0.65 and ((fq_sum < 0.00085 and fq_min < 0.00025) or fq_min < 0.00012):
            self.empty_grasp_events += 1
            if self.grasp_retries < self.max_grasp_retries:
                self.grasp_retries += 1
                self.retract_steps = 30
                # Lift straight back to P1 entry pregrasp posture (above holder, NEVER swinging across dock!)
                self.retract_q_target = self.p1_entry_q.copy() if self.p1_entry_q is not None else q_cur.copy()
                self.transitions.append(dict(
                    step=int(step), from_phase=int(self.phase), to_phase=0,
                    reason=f"empty_grasp_abort_fqsum_{fq_sum*1000:.2f}mm",
                    tcp=[round(float(x), 4) for x in raw14[:3]], grip=round(grip, 4),
                ))
                self.phase = 0
                self.settle_steps = 0
                self.min_grasp_z = float("inf")
                cmd = np.zeros(8, dtype=np.float32)
                cmd[:7] = q_cur
                cmd[7] = 0.0
                return self.phase, cmd, True, False

        # Open-gripper stall recovery in P2: if P2 rose >= 20 mm above grasp height WITHOUT
        # closing gripper (grip < 0.15) for 25 steps, re-descend via P1 (pure proprioception!).
        if self.phase == 2 and grip < 0.15 and getattr(self, "min_grasp_z", 99.0) < 10.0 and (float(raw14[2]) - float(self.min_grasp_z)) >= 0.020:
            self.open_lift_steps += 1
            if self.open_lift_steps >= 25 and self.grasp_retries < self.max_grasp_retries:
                self.grasp_retries += 1
                self.open_lift_steps = 0
                self.phase = 1
                self.settle_steps = 0
                self.transitions.append(dict(
                    step=int(step), from_phase=2, to_phase=1,
                    reason=f"open_lift_retry_{self.grasp_retries}",
                    tcp=[round(float(x), 4) for x in raw14[:3]], grip=round(grip, 3),
                ))
                cmd = np.zeros(8, dtype=np.float32)
                cmd[:7] = q_cur
                cmd[7] = 0.0
                return self.phase, cmd, True, False
        else:
            self.open_lift_steps = 0

        # Never advance P2->P3 unless real finger gap confirms BOTH fingers firmly hold USB
        block_p2 = (self.phase == 2 and (fq_sum < 0.00110 or fq_min < 0.00028))
        # Never advance P4->P5 unless TCP genuinely descended >= 14 mm from P4 entry AND
        # sits within +5.5 mm of min_grasp_z (prevents premature P4->P5 pullout at dep=8.5-9.9mm!)
        block_p4 = False
        if self.phase == 4:
            if self.p4_entry_z is None:
                self.p4_entry_z = float(raw14[2])
            if (self.p4_entry_z - float(raw14[2])) < 0.014:
                block_p4 = True
            if self.min_grasp_z < 10.0 and (float(raw14[2]) - self.min_grasp_z) > 0.0058:
                block_p4 = True
        return None, None, block_p2, block_p4


class SafePhase12SwitcherV2(NonPrivilegedPhase12Switcher, SafeFingerGuardMixin):
    """Non-privileged 12-phase switcher calibrated for both EXP12 and v3_real WideGraspTeacher envelopes."""

    def __init__(self, start_phase: int = 0, max_grasp_retries: int = 2) -> None:
        super().__init__(start_phase=start_phase)
        self.init_guard(max_grasp_retries=max_grasp_retries)
        self.p3_hover_settle = 0
        self.p4_zslip_settle = 0
        self.p7_bottom_settle = 0
        self.hist_z: list[float] = []
        self.hist_tcp: list[np.ndarray] = []

    def _record_motion_hist(self, st21: np.ndarray) -> tuple[float, float, float]:
        tcp = np.asarray(st21[:3], dtype=np.float64)
        q = np.asarray(st21[6:13], dtype=np.float64)
        self.hist_q.append(q.copy())
        if len(self.hist_q) > 10:
            self.hist_q.pop(0)
        dq5 = float(np.linalg.norm(q - self.hist_q[-6])) if len(self.hist_q) >= 6 else 1.0
        self.hist_z.append(float(tcp[2]))
        if len(self.hist_z) > 10:
            self.hist_z.pop(0)
        dz5 = abs(float(tcp[2]) - self.hist_z[-6]) if len(self.hist_z) >= 6 else 1.0
        self.hist_tcp.append(tcp.copy())
        if len(self.hist_tcp) > 10:
            self.hist_tcp.pop(0)
        dtcp5 = float(np.linalg.norm(tcp - self.hist_tcp[-6])) if len(self.hist_tcp) >= 6 else 1.0
        return dq5, dz5, dtcp5

    def _update_calibrated(self, step: int, st21: np.ndarray, fq_sum: float, fq_min: float, dq5: float, dz5: float, dtcp5: float) -> int:
        tcp = st21[:3]
        grip = float(st21[13])

        if self.phase == 0:
            if tcp[2] >= 0.815:
                self.saw_raise_high = True
            above_holder = (
                (0.338 <= tcp[0] <= 0.410)
                and (-0.300 <= tcp[1] <= -0.232)
                and (0.814 <= tcp[2] <= 0.850)
            )
            if self.saw_raise_high and grip < 0.05 and above_holder and dq5 < 0.055:
                self.settle_steps += 1
            else:
                self.settle_steps = 0
            if self.settle_steps >= 4:
                self._advance(step, 1, "arrived_pregrasp", st21)

        elif self.phase == 1:
            in_holder = (
                (0.340 <= tcp[0] <= 0.405)
                and (-0.295 <= tcp[1] <= -0.235)
                and (tcp[2] <= 0.8055)
            )
            if in_holder and grip >= 0.25:
                self.min_grasp_z = float(tcp[2])
                self._advance(step, 2, "arrived_grasp_closing", st21)
            elif in_holder and dq5 < 0.006:
                self.settle_steps += 1
                if self.settle_steps >= 6:
                    self.min_grasp_z = float(tcp[2])
                    self._advance(step, 2, "arrived_grasp_height", st21)
            else:
                self.settle_steps = 0

        elif self.phase == 2:
            self.min_grasp_z = min(self.min_grasp_z, float(tcp[2]))
            if grip >= 0.80 and fq_sum >= 0.00110 and fq_min >= 0.00028 and (float(tcp[2]) - self.min_grasp_z) >= 0.027 and dq5 < 0.006:
                self.settle_steps += 1
            else:
                self.settle_steps = 0
            if self.settle_steps >= 8:
                self._advance(step, 3, "grasped_and_lifted", st21)
            elif grip >= 0.80 and fq_sum >= 0.00110 and fq_min >= 0.00028 and (float(tcp[2]) - self.min_grasp_z) >= 0.075:
                self._advance(step, 3, "grasped_and_lifted_overshoot_75mm", st21)

        elif self.phase == 3:
            in_mouth_box = (
                grip >= 0.80
                and fq_sum >= 0.00105
                and (0.372 <= tcp[0] <= 0.415)
                and (-0.175 <= tcp[1] <= -0.132)
                and (tcp[2] <= 0.8265)
            )
            if in_mouth_box:
                self._advance(step, 4, "arrived_port_mouth", st21)
                self.p3_hover_settle = 0
            else:
                over_dock = (
                    grip >= 0.80
                    and fq_sum >= 0.00105
                    and (0.370 <= tcp[0] <= 0.418)
                    and (-0.178 <= tcp[1] <= -0.130)
                    and (tcp[2] <= 0.8540)
                    and dq5 < 0.005
                )
                self.p3_hover_settle = (self.p3_hover_settle + 1) if over_dock else 0
                if self.p3_hover_settle >= 35:
                    self._advance(step, 4, "arrived_port_hover_settled", st21)
                    self.p3_hover_settle = 0

        elif self.phase == 4:
            # Guarded above by p4_entry_z - tcp_z >= 14mm AND tcp_z - min_grasp_z <= +5.8mm (non-privileged!).
            # Do not count settle while TCP is still descending (z_drop5 > 0.08 mm/5 steps); hold 33 steps (1.10s > 1.0s).
            z_drop5 = (self.hist_z[-6] - float(tcp[2])) if len(self.hist_z) >= 6 else 1.0
            if (
                grip >= 0.80
                and fq_sum >= 0.00105
                and (0.365 <= tcp[0] <= 0.420)
                and (-0.185 <= tcp[1] <= -0.130)
                and (0.7945 <= tcp[2] <= 0.8048)
                and z_drop5 <= (0.00008 if float(tcp[2]) > 0.8006 else 0.00025)
                and (dq5 < 0.0035 or (dtcp5 < 0.00070 and dz5 < 0.00030))
            ):
                self.settle_steps += 1
            else:
                self.settle_steps = 0
            if self.settle_steps >= 33:
                self.min_dock_z = float(tcp[2])
                self.p4_zslip_settle = 0
                self._advance(step, 5, "seated_in_dock", st21)
            else:
                zslip_ok = (
                    grip >= 0.80
                    and fq_sum >= 0.00105
                    and (0.365 <= tcp[0] <= 0.420)
                    and (-0.185 <= tcp[1] <= -0.130)
                    and (0.7940 <= tcp[2] <= 0.8054)
                    and (self.min_grasp_z >= 10.0 or (float(tcp[2]) - self.min_grasp_z) <= 0.0045)
                    and (self.p4_entry_z is not None and (self.p4_entry_z - float(tcp[2])) >= 0.018)
                    and z_drop5 <= 0.00025
                    and (dq5 < 0.0040 or dtcp5 < 0.00080)
                    and dz5 < 0.00035
                )
                self.p4_zslip_settle = (self.p4_zslip_settle + 1) if zslip_ok else 0
                if self.p4_zslip_settle >= 75:
                    self.min_dock_z = float(tcp[2])
                    self.p4_zslip_settle = 0
                    self._advance(step, 5, "seated_in_dock_zslip", st21)

        elif self.phase == 5:
            self.min_dock_z = min(self.min_dock_z, float(tcp[2]))
            if grip >= 0.80 and (float(tcp[2]) - self.min_dock_z) >= 0.036 and dq5 < 0.006:
                self.settle_steps += 1
            else:
                self.settle_steps = 0
            if self.settle_steps >= 8:
                self._advance(step, 6, "withdrawn_clear", st21)

        elif self.phase == 6:
            above_holder = (
                (0.335 <= tcp[0] <= 0.415)
                and (-0.305 <= tcp[1] <= -0.230)
                and (tcp[2] <= 0.8420)
            )
            if grip >= 0.80 and above_holder and dq5 < 0.012:
                self.settle_steps += 1
            else:
                self.settle_steps = 0
            if self.settle_steps >= 5:
                self._advance(step, 7, "returned_above_holder", st21)

        elif self.phase == 7:
            z_rel_ok = (
                (-0.0063 <= float(tcp[2]) - self.min_grasp_z <= 0.0052)
                if self.min_grasp_z < 10.0
                else (float(tcp[2]) <= 0.8045)
            )
            in_holder_xy = (0.345 <= float(tcp[0]) <= 0.406) and (-0.295 <= float(tcp[1]) <= -0.236)
            in_holder_z = (0.7900 <= float(tcp[2]) <= 0.8060)
            seated_basic = (grip >= 0.80 and fq_sum >= 0.00105 and in_holder_xy and in_holder_z and z_rel_ok and dq5 < 0.0025)
            if seated_basic:
                self.settle_steps += 1
            else:
                self.settle_steps = 0
            if self.settle_steps >= 12:
                self._advance(step, 8, "seated_in_holder", st21)
                self.p7_bottom_settle = 0
            else:
                seated_floor = (
                    grip >= 0.80
                    and fq_sum >= 0.00105
                    and in_holder_xy
                    and in_holder_z
                    and z_rel_ok
                    and dz5 < 0.00035
                    and dtcp5 < 0.00075
                )
                self.p7_bottom_settle = (self.p7_bottom_settle + 1) if seated_floor else 0
                if self.p7_bottom_settle >= 14:
                    self._advance(step, 8, "seated_in_holder_floor", st21)
                    self.p7_bottom_settle = 0
                elif grip >= 0.80 and in_holder_xy and in_holder_z and z_rel_ok and fq_min < 0.00010 and dz5 < 0.00050:
                    self.p7_fq_settle = getattr(self, "p7_fq_settle", 0) + 1
                    if self.p7_fq_settle >= 5:
                        self._advance(step, 8, "seated_in_holder_fq_release", st21)
                        self.p7_bottom_settle = 0
                        self.p7_fq_settle = 0
                else:
                    self.p7_fq_settle = 0

        elif self.phase == 8:
            if grip <= 0.04 and dq5 < 0.004:
                self.open_steps += 1
            else:
                self.open_steps = 0
            if self.open_steps >= 36:
                self.min_clear_z = float(tcp[2])
                self._advance(step, 9, "released_in_holder", st21)

        elif self.phase == 9:
            self.min_clear_z = min(self.min_clear_z, float(tcp[2]))
            if grip <= 0.05 and (float(tcp[2]) - self.min_clear_z) >= 0.026 and dq5 < 0.006:
                self.settle_steps += 1
            else:
                self.settle_steps = 0
            if self.settle_steps >= 6:
                self._advance(step, 10, "cleared_holder", st21)

        elif self.phase == 10:
            herr = float(np.max(np.abs(st21[14:21])))
            if grip <= 0.05 and herr <= 0.16:
                self._advance(step, 11, "entered_home_brake_basin", st21)

        return self.phase

    def step_update(self, step: int, st21: np.ndarray, fq_margin: np.ndarray, q_home0: np.ndarray):
        dq5, dz5, dtcp5 = self._record_motion_hist(st21)
        ph_ov, cmd_ov, block_p2, block_p4 = self.guard_step(step, st21[:14], fq_margin, q_home0)
        if cmd_ov is not None:
            return ph_ov, cmd_ov
        if block_p2 or block_p4:
            self.settle_steps = 0
            self.p3_hover_settle = 0
            return self.phase, None
        prev = self.phase
        fq_sum = float(np.sum(fq_margin))
        fq_min = float(np.min(fq_margin))
        new_p = self._update_calibrated(step, st21, fq_sum, fq_min, dq5, dz5, dtcp5)
        if prev != 4 and new_p == 4:
            self.p4_entry_z = float(st21[2])
        return new_p, None


class SafePhase13SwitcherV2(NonPrivilegedPhase13Switcher, SafeFingerGuardMixin):
    def __init__(self, start_phase: int = 0, tcp_grasp_mem: np.ndarray | None = None, max_grasp_retries: int = 2) -> None:
        super().__init__(start_phase=start_phase, tcp_grasp_mem=tcp_grasp_mem)
        self.init_guard(max_grasp_retries=max_grasp_retries)

    def step_update(self, step: int, raw_st: np.ndarray, fq_margin: np.ndarray, q_home0: np.ndarray):
        ph_ov, cmd_ov, block_p2, block_p4 = self.guard_step(step, raw_st[:14], fq_margin, q_home0)
        if cmd_ov is not None:
            return ph_ov, cmd_ov
        if block_p2 or block_p4:
            self.settle_steps = 0
            return self.phase, None
        prev = self.phase
        new_p = super().update(step, raw_st, q_home0)
        if prev != 4 and new_p == 4:
            self.p4_entry_z = float(raw_st[2])
        return new_p, None


def load_model_fp32(cfg_dir: Path, config_name: str, ckpt: str):
    with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.1"):
        cfg = compose(config_name=config_name)
    inp, out = build_openpi_transforms(
        cfg.actor.model.model_path,
        cfg.actor.model.openpi.config_name,
        dict(cfg.actor.model.openpi_data),
    )
    tx, un = T.compose(inp), T.compose(out)
    model = get_model(cfg.actor.model, torch_dtype=torch.float32).eval()
    state = {"fsdp_checkpoint": {"model": model.state_dict()}}
    dcp.load(state, checkpoint_id=ckpt)
    model.load_state_dict(state["fsdp_checkpoint"]["model"], strict=True)
    del state
    model = model.cuda()
    return model, tx, un


def infer_chunk(model, tx, un, raw: dict, step_i: int, flow_m: int, stream: torch.cuda.Stream) -> np.ndarray:
    processed = tx(raw)
    batch = model._observation_to_device(
        default_collate([{k: v for k, v in processed.items() if k != "actions"}])
    )
    torch.manual_seed(NOISE + step_i)
    if flow_m > 1:
        with torch.inference_mode(), torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16):
            obs_prep = openpi_model.preprocess_observation(batch, train=False)
            _, prefix_mask, kv_cache = model.build_prefix_cache(obs_prep)
            kv_exp = tuple(tuple(t.expand(flow_m, *t.shape[1:]) for t in layer) for layer in kv_cache)
            pm_exp = prefix_mask.expand(flow_m, -1)
            obs_k = dataclasses.replace(obs_prep, state=obs_prep.state.expand(flow_m, -1))
            x_t = torch.randn((flow_m, 16, 32), device=obs_prep.state.device, dtype=torch.float32)
            dt = -1.0 / 5.0
            t_val = 1.0
            while t_val >= -dt / 2:
                t_tensor = torch.full((flow_m,), t_val, device=obs_prep.state.device, dtype=torch.float32)
                sout = model.run_suffix(obs_k, x_t, t_tensor, kv_exp, pm_exp)
                vt = model.velocity_from_suffix(sout)
                x_t = x_t + dt * vt.float()
                t_val += dt
            pred_np = x_t.mean(dim=0).float().cpu().numpy()
        stream.synchronize()
        actions = un(dict(actions=pred_np, state=processed["state"]))["actions"]
    else:
        with torch.inference_mode(), torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.sample_actions(batch, num_steps=5)
        stream.synchronize()
        actions = un(dict(actions=pred[0].float().cpu().numpy(), state=processed["state"]))["actions"]
    return actions


def select_v2_rows(split: str, seeds: list[int] | None, limit: int, scenes_root: Path):
    manifest = json.loads((EXP13 / f"{split}_manifest.json").read_text())
    full = [r for r in manifest if r["record_kind"] == "full_cycle"]
    scenes_v2 = json.loads((scenes_root / (f"stage2_{split}_scenes_v3.json" if (scenes_root / f"stage2_{split}_scenes_v3.json").exists() else f"stage2_{split}_scenes_v2.json")).read_text())
    by_seed = {int(s["seed"]): s for s in scenes_v2}
    full = [r for r in full if int(r["parent_scene_seed"]) in by_seed]
    if seeds:
        order = {s: i for i, s in enumerate(seeds)}
        full = [r for r in full if int(r["parent_scene_seed"]) in order]
        full.sort(key=lambda r: order[int(r["parent_scene_seed"])])
    elif limit < len(full):
        idxs = np.linspace(0, len(full) - 1, min(limit, len(full)), dtype=int)
        full = [full[i] for i in idxs]
    return [dict(scene=dict(by_seed[int(r["parent_scene_seed"])]), record=r) for r in full]


def run_eval(args):
    usb12.register()
    usb13.register()
    ckpt12 = str(EXP12 / "training/phase12_sft8/phase12_sft8/checkpoints/global_step_3000/actor/dcp_checkpoint")
    ckpt_p8 = str(EXP13 / "training/p8_vision_sft8/p8_vision_sft8/checkpoints/global_step_1500/actor/dcp_checkpoint")
    ckpt13 = str(EXP13 / "training/phase13_sft8/phase13_sft8/checkpoints/global_step_2500/actor/dcp_checkpoint")

    models = {}
    if args.mode == "tri_safe":
        models["p12"] = load_model_fp32(EXP12 / "configs", "phase12_sft8", ckpt12)
        models["p8"] = load_model_fp32(EXP13 / "configs", "p8_vision_sft8", ckpt_p8)
        models["p13"] = load_model_fp32(EXP13 / "configs", "phase13_sft8", ckpt13)
    elif args.mode in ("single_p12_safe", "single_p12_p4_only"):
        models["p12"] = load_model_fp32(EXP12 / "configs", "phase12_sft8", ckpt12)
    else:
        raise ValueError(args.mode)

    flow_m_map = parse_phase_int_map(args.phase_flow_m)
    k_override = parse_phase_int_map(args.phase_exec_k)
    stream = torch.cuda.Stream(priority=0)
    scenes_root = Path(args.scenes_root)
    dest = H2 / "eval" / args.label
    dest.mkdir(parents=True, exist_ok=True)
    rows = select_v2_rows(args.split, args.seeds, args.limit, scenes_root)

    reports = []
    t_all = time.time()
    for row in rows:
        t0 = time.time()
        sc = row["scene"]
        seed = int(sc["seed"])
        rec_path = Path(row["record"]["path"])
        snap = rec_path / args.start_snapshot
        run_dir = dest / str(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        env = PolicyExecutor(Path(sc["scene"]), run_dir, seed, brightness=float(sc["brightness"]))
        try:
            env.reset(snapshot=snap)
            task = env._task
            obs0 = env.observe()
            if args.start_phase == 0:
                q_home0 = np.asarray(obs0["observation.state"][6:13], dtype=np.float32).copy()
            else:
                with np.load(rec_path / "demonstration_full.npz", allow_pickle=True) as d:
                    q_home0 = d["observation.state"][0, 6:13].astype(np.float32).copy()
            init_mem = None
            if args.start_phase >= 3:
                with np.load(rec_path / "demonstration_full.npz", allow_pickle=True) as d:
                    _, init_mem = usb13.capture_grasp_mem_from_demo(d["observation.state"])

            if args.mode == "tri_safe":
                sw = SafePhase13SwitcherV2(start_phase=args.start_phase, tcp_grasp_mem=init_mem)
                k_map = dict(P13_EXEC_STEPS)
            else:
                sw = SafePhase12SwitcherV2(start_phase=args.start_phase)
                k_map = dict(P12_EXEC_STEPS)
            k_map.update(k_override)

            pending = []
            cur_phase = args.start_phase
            min_lat = float("inf")
            max_dep = float("-inf")
            min_holder_dz = float("inf")
            min_home_err = float("inf")
            step_done = 0

            for i in range(args.max_steps):
                step_done = i + 1
                obs = env.observe()
                raw_st = obs["observation.state"]
                fq_margin = np.asarray(task.d.qpos[task.fq] - task.finger_lower, dtype=np.float32)

                if args.mode == "tri_safe":
                    new_phase, cmd_ov = sw.step_update(i, raw_st, fq_margin, q_home0)
                else:
                    st21 = usb12.make_state21(raw_st, q_home0)
                    new_phase, cmd_ov = sw.step_update(i, st21, fq_margin, q_home0)

                if new_phase != cur_phase:
                    cur_phase = new_phase
                    pending.clear()

                tip, _ = task.object_pose()
                d_port = tip - task.port
                dep = float(np.dot(d_port, task.S[:, 2]) * 1000.0)
                lat = float(np.linalg.norm(d_port - np.dot(d_port, task.S[:, 2]) * task.S[:, 2]) * 1000.0)
                min_lat = min(min_lat, lat)
                max_dep = max(max_dep, dep)
                if cur_phase in (7, 8):
                    min_holder_dz = min(min_holder_dz, float((tip[2] - task.initial_tip[2]) * 1000.0))
                if cur_phase >= 10:
                    min_home_err = min(min_home_err, float(np.max(np.abs(raw_st[6:13] - q_home0))))

                if cmd_ov is not None:
                    cmd = cmd_ov
                    pending.clear()
                else:
                    if not pending:
                        if args.mode == "tri_safe":
                            if cur_phase <= 5:
                                m, tx, un = models["p12"]
                                st = usb12.make_state21(raw_st, q_home0)
                                prompt = usb12.PHASE_PROMPTS[cur_phase]
                            elif cur_phase in (7, 8):
                                m, tx, un = models["p8"]
                                st = usb12.make_state21(raw_st, q_home0)
                                prompt = usb12.PHASE_PROMPTS[7]
                            else:
                                m, tx, un = models["p13"]
                                st = usb13.make_state24(raw_st, q_home0, sw.tcp_grasp_mem)
                                prompt = usb13.PHASE_PROMPTS[cur_phase]
                        else:
                            m, tx, un = models["p12"]
                            st = usb12.make_state21(raw_st, q_home0)
                            prompt = usb12.PHASE_PROMPTS[cur_phase]
                        raw_in = dict(
                            state=st,
                            image=obs["observation.images.front"],
                            left_wrist_image=obs["observation.images.left_wrist"],
                            wrist_image=obs["observation.images.wrist"],
                            prompt=prompt,
                        )
                        pred = infer_chunk(m, tx, un, raw_in, i, int(flow_m_map.get(cur_phase, 1)), stream)
                        pending = list(pred[: k_map[cur_phase]])
                    cmd = np.asarray(pending.pop(0), dtype=np.float32).copy()
                    # Hard real-encoder finger safety clamp before lower-0.00015
                    if cmd[7] >= 0.85 and float(np.min(fq_margin)) < 0.00018 and float(np.sum(fq_margin)) < 0.00085:
                        cmd[7] = 0.0

                res = env.step(cmd)
                if args.mode == "single_p12_p4_only" and "insert" in env.outcomes()["physical"].get("event_times", {}):
                    break
                if res["terminated"] or res["truncated"]:
                    break

            out = env.outcomes()
            phys = out["physical"]
            rep = dict(
                seed=seed, mode=args.mode, start_phase=args.start_phase, steps=step_done,
                elapsed_s=round(time.time() - t0, 1),
                success=bool(phys.get("success", False)),
                events=phys.get("event_times", {}),
                min_lateral_mm=round(min_lat, 3) if np.isfinite(min_lat) else None,
                max_depth_mm=round(max_dep, 3) if np.isfinite(max_dep) else None,
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
            env.close()

    ev_names = ("lift", "insert", "withdraw", "return", "release", "home")
    counts = {e: sum(1 for r in reports if e in r["events"]) for e in ev_names}
    summary = dict(
        completed=True, label=args.label, mode=args.mode, split=args.split,
        total=len(reports), full_success=sum(1 for r in reports if r["success"]),
        safety_stops=sum(1 for r in reports if r["invalid_reason"] == "safety_stop"),
        event_counts=counts, elapsed_total_s=round(time.time() - t_all, 1), reports=reports,
    )
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY %s: full=%d/%d safety=%d events=%s time=%.1fs" % (
        args.label, summary["full_success"], summary["total"], summary["safety_stops"], counts, summary["elapsed_total_s"]
    ), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("tri_safe", "single_p12_safe", "single_p12_p4_only"), required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--scenes-root", default=str(H2 / "scenes_v2_keepvis"))
    ap.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--start-snapshot", default="cycle_phase_raise_to_pregrasp.npz")
    ap.add_argument("--start-phase", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--phase-flow-m", default="4:8")
    ap.add_argument("--phase-exec-k", default="4:2,7:2,8:2")
    run_eval(ap.parse_args())
