# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Strictly robot-only contact proprioception and analytical Cartesian-to-joint residual mapping.

ZERO privileged object/socket/holder pose is ever read. All signals correspond 1-to-1 to
real Unitree G1 EDU+ + Dex1-1 SDK2 encoders, motor torques, and robot URDF kinematics.
"""
from __future__ import annotations

import mujoco
import numpy as np

ARM_DOF_SLICE = slice(24, 31)
ARM_ACT_SLICE = slice(24, 31)
CONTACT_PROPRIO_DIM = 44


def extract_contact_proprio(
    task,
    raw_state: np.ndarray,
    last_cmd_q7: np.ndarray,
    q_prev4: np.ndarray,
    tcp_entry: np.ndarray,
    stall_steps: int,
    last_res_act6: np.ndarray,
) -> np.ndarray:
    """Build 44D strictly robot-only contact & relative proprioception vector."""
    raw14 = np.asarray(raw_state, dtype=np.float32)[:14]
    tcp = raw14[:3]
    rotvec = raw14[3:6]
    q_right = raw14[6:13]
    fq_margin = np.asarray(task.d.qpos[task.fq] - task.finger_lower, dtype=np.float32)
    q_err = (np.asarray(last_cmd_q7, dtype=np.float32) - q_right) * 100.0
    tau_act = np.asarray(task.d.actuator_force[ARM_ACT_SLICE], dtype=np.float32) / 10.0
    dq_recent = (q_right - np.asarray(q_prev4, dtype=np.float32)) * 50.0
    delta_tcp = (tcp - np.asarray(tcp_entry, dtype=np.float32)) * 100.0
    stall_norm = np.array([min(float(stall_steps) / 30.0, 2.0)], dtype=np.float32)
    res6 = np.asarray(last_res_act6, dtype=np.float32).reshape(6)

    feat = np.concatenate([
        np.array([(float(tcp[2]) - 0.81) * 20.0], dtype=np.float32),  # 1
        rotvec,                                                        # 3
        q_right,                                                       # 7
        np.clip(fq_margin * 500.0, -2.0, 10.0),                        # 2
        np.clip(q_err, -5.0, 5.0),                                     # 7
        np.clip(tau_act, -5.0, 5.0),                                   # 7
        np.clip(dq_recent, -5.0, 5.0),                                 # 7
        np.clip(delta_tcp, -5.0, 5.0),                                 # 3
        stall_norm,                                                    # 1
        np.clip(res6, -1.0, 1.0),                                      # 6
    ], axis=0).astype(np.float32)
    assert feat.shape == (CONTACT_PROPRIO_DIM,) and np.isfinite(feat).all()
    return feat


def compute_robot_tcp_jacobian(task) -> np.ndarray:
    """Compute exact 6x7 right-arm TCP Jacobian from robot-only model (t.rm, t.rd, t.rtool)."""
    task.robot_forward()
    jacp = np.zeros((3, task.rm.nv), dtype=np.float64)
    jacr = np.zeros((3, task.rm.nv), dtype=np.float64)
    mujoco.mj_jacSite(task.rm, task.rd, jacp, jacr, task.rtool)
    return np.vstack([jacp[:, ARM_DOF_SLICE], jacr[:, ARM_DOF_SLICE]])  # (6, 7)


def apply_cartesian_residual_to_chunk(
    task,
    ref_chunk: np.ndarray,
    res_act6: np.ndarray,
    *,
    scale_xyz_m: float = 0.0020,
    scale_rot_rad: float = 0.020,
    max_dq_rad: float = 0.015,
    damping: float = 1e-2,
) -> np.ndarray:
    """Map bounded res_act6 in [-1, 1]^6 via robot-only J_pinv(q) onto ref_chunk[:, :7].

    Gripper channel ref_chunk[:, 7] is preserved verbatim so residual RL never drops the USB.
    """
    a = np.asarray(ref_chunk, dtype=np.float32).copy()
    r = np.clip(np.asarray(res_act6, dtype=np.float64).reshape(6), -1.0, 1.0)
    if float(np.max(np.abs(r))) < 1e-8:
        return a
    dx = np.empty(6, dtype=np.float64)
    dx[:3] = r[:3] * float(scale_xyz_m)
    dx[3:] = r[3:] * float(scale_rot_rad)
    J = compute_robot_tcp_jacobian(task)
    J_pinv = J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(6))
    dq = np.clip(J_pinv @ dx, -max_dq_rad, max_dq_rad).astype(np.float32)
    a[:, :7] = a[:, :7] + dq[None, :]
    return a
