#!/usr/bin/env python3
"""Retarget AMASS SMPL motion to Unitree H1 via MuJoCo FK + optimization.

Uses MuJoCo to load H1 model for accurate FK. Optimizes H1 DOF angles
to match SMPL joint positions per frame. Simple and self-contained.

Usage:
    python retarget_smpl_to_h1.py --file /path/to/amass.npz --output h1_motion.npz
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from scipy.optimize import minimize
import mujoco

# ── SMPL ──────────────────────────────────────────────────────────────────────
SMPL_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]
SMPL_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]

# Approximate bone offsets in SMPL frame (Y=up, Z=forward, X=right)
SMPL_OFFSETS = np.zeros((24, 3))
SMPL_OFFSETS[1]  = [ 0.065,-0.087,-0.030]; SMPL_OFFSETS[2]  = [-0.065,-0.087,-0.030]
SMPL_OFFSETS[3]  = [ 0.000, 0.120, 0.010]; SMPL_OFFSETS[4]  = [ 0.000,-0.400, 0.000]
SMPL_OFFSETS[5]  = [ 0.000,-0.400, 0.000]; SMPL_OFFSETS[6]  = [ 0.000, 0.130, 0.000]
SMPL_OFFSETS[7]  = [ 0.000,-0.420, 0.000]; SMPL_OFFSETS[8]  = [ 0.000,-0.420, 0.000]
SMPL_OFFSETS[9]  = [ 0.000, 0.180, 0.000]; SMPL_OFFSETS[10] = [ 0.000,-0.060, 0.120]
SMPL_OFFSETS[11] = [ 0.000,-0.060, 0.120]; SMPL_OFFSETS[12] = [ 0.000, 0.130, 0.020]
SMPL_OFFSETS[13] = [ 0.070, 0.060,-0.010]; SMPL_OFFSETS[14] = [-0.070, 0.060,-0.010]
SMPL_OFFSETS[15] = [ 0.000, 0.120, 0.020]; SMPL_OFFSETS[16] = [ 0.120, 0.000, 0.000]
SMPL_OFFSETS[17] = [-0.120, 0.000, 0.000]; SMPL_OFFSETS[18] = [ 0.260, 0.000, 0.000]
SMPL_OFFSETS[19] = [-0.260, 0.000, 0.000]; SMPL_OFFSETS[20] = [ 0.260, 0.000, 0.000]
SMPL_OFFSETS[21] = [-0.260, 0.000, 0.000]; SMPL_OFFSETS[22] = [ 0.100, 0.000, 0.000]
SMPL_OFFSETS[23] = [-0.100, 0.000, 0.000]

# Robot ↔ SMPL joint matches for position fitting
JOINT_MATCHES = [
    ("left_hip_pitch_link", "L_Hip"),
    ("left_knee_link", "L_Knee"),
    ("left_ankle_link", "L_Ankle"),
    ("right_hip_pitch_link", "R_Hip"),
    ("right_knee_link", "R_Knee"),
    ("right_ankle_link", "R_Ankle"),
    ("torso_link", "Spine3"),
    ("left_shoulder_roll_link", "L_Shoulder"),
    ("left_elbow_link", "L_Elbow"),
    ("right_shoulder_roll_link", "R_Shoulder"),
    ("right_elbow_link", "R_Elbow"),
]

def smpl_fk(poses_aa, trans):
    """SMPL FK: (72,) aa + (3,) trans → (24, 3) positions."""
    rots = [sRot.from_rotvec(poses_aa[j*3:(j+1)*3]).as_matrix() for j in range(24)]
    g_rot, g_pos = np.zeros((24,3,3)), np.zeros((24,3))
    g_rot[0], g_pos[0] = rots[0], trans
    for j in range(1, 24):
        p = SMPL_PARENTS[j]
        g_rot[j] = g_rot[p] @ rots[j]
        g_pos[j] = g_pos[p] + g_rot[p] @ SMPL_OFFSETS[j]
    return g_pos

def smpl_to_h1_coords(pos):
    """Convert positions from SMPL frame (Y-up, Z-fwd) to H1 frame (Z-up, X-fwd)."""
    out = np.zeros_like(pos)
    out[..., 0] =  pos[..., 2]   # H1 X = SMPL Z (forward)
    out[..., 1] = -pos[..., 0]   # H1 Y = -SMPL X (left)
    out[..., 2] =  pos[..., 1]   # H1 Z = SMPL Y (up)
    return out

# Position mapping matrix: SMPL→H1 (same as smpl_to_h1_coords)
# H1_X = SMPL_Z, H1_Y = -SMPL_X, H1_Z = SMPL_Y
P_SMPL_TO_H1 = np.array([
    [0, 0, 1],   # H1 X = SMPL Z
    [-1, 0, 0],  # H1 Y = -SMPL X
    [0, 1, 0],   # H1 Z = SMPL Y
], dtype=float)

def smpl_root_to_h1(smpl_root_aa):
    """Convert SMPL root rotation to H1 root quaternion [w,x,y,z].

    Extracts only the heading (yaw) from SMPL root rotation and converts
    it to H1 frame. Body tilt is handled by the DOF optimizer.
    This avoids issues with the improper (det=-1) frame change matrix.
    """
    R_smpl = sRot.from_rotvec(smpl_root_aa)

    # Extract heading: project the SMPL forward direction onto the ground plane
    # SMPL forward = +Z axis
    fwd_smpl = R_smpl.apply([0, 0, 1])  # where forward points after rotation

    # Heading angle in SMPL: angle of forward projected onto XZ plane
    yaw_smpl = np.arctan2(-fwd_smpl[0], fwd_smpl[2])  # angle from +Z toward -X

    # Convert to H1: SMPL yaw around Y → H1 yaw around Z
    # In H1, yaw is rotation around Z. SMPL forward (+Z) maps to H1 forward (+X).
    # SMPL left turn (positive yaw around Y) = H1 left turn (positive yaw around Z)...
    # but our mapping flips handedness (SMPL X(right) → H1 -Y(left))
    # So the heading angle stays the same magnitude but may flip sign.
    # Actually: atan2(-SMPL_X, SMPL_Z) in SMPL frame
    #         = atan2(H1_Y, H1_X) in H1 frame (using our mapping)
    # This is the standard heading in H1's XY plane.
    yaw_h1 = yaw_smpl

    R_h1 = sRot.from_euler('z', yaw_h1)
    q = R_h1.as_quat()  # [x,y,z,w]
    return np.array([q[3], q[0], q[1], q[2]])  # [w,x,y,z]


class H1:
    def __init__(self, xml_path):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)

        # DOF info (skip free joint 0)
        self.dof_names, self.dof_qidx = [], []
        for i in range(1, self.m.njnt):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.dof_names.append(name)
                self.dof_qidx.append(self.m.jnt_qposadr[i])
        self.ndof = len(self.dof_names)
        self.lo = np.array([self.m.jnt_range[i+1, 0] for i in range(self.ndof)])
        self.hi = np.array([self.m.jnt_range[i+1, 1] for i in range(self.ndof)])

        # Body names
        self.body_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or f"b{i}"
                           for i in range(self.m.nbody)]

        # Match indices
        self.h1_idx, self.smpl_idx = [], []
        for h1n, sn in JOINT_MATCHES:
            hi = self.body_names.index(h1n) if h1n in self.body_names else -1
            si = SMPL_NAMES.index(sn) if sn in SMPL_NAMES else -1
            if hi >= 0 and si >= 0:
                self.h1_idx.append(hi)
                self.smpl_idx.append(si)
        print(f"H1: {self.ndof} DOFs, {len(self.h1_idx)} matches")

    def fk(self, dof, root_pos, root_quat_wxyz):
        """Set state and compute FK. Returns body xpos."""
        self.d.qpos[:] = 0
        self.d.qpos[0:3] = root_pos
        self.d.qpos[3:7] = root_quat_wxyz
        for i, qi in enumerate(self.dof_qidx):
            self.d.qpos[qi] = dof[i]
        mujoco.mj_forward(self.m, self.d)
        return self.d.xpos.copy(), self.d.xquat.copy()


def fit_frame(h1, smpl_pos_h1, root_pos, root_quat, prev_dof):
    """Optimize DOFs to minimize position error between H1 FK and SMPL."""
    # Target: SMPL joint positions relative to root (pelvis)
    tgt = smpl_pos_h1[h1.smpl_idx] - smpl_pos_h1[0:1]  # (N_match, 3)

    def cost(dof):
        xpos, _ = h1.fk(dof, root_pos, root_quat)
        h1_pos = xpos[h1.h1_idx]
        # Pelvis body index is 1 (0=world)
        h1_rel = h1_pos - xpos[1:2]
        return float(np.sum((h1_rel - tgt) ** 2)) + 0.0005 * float(np.sum(dof**2))

    bounds = list(zip(h1.lo, h1.hi))
    res = minimize(cost, prev_dof, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-12})
    return res.x, res.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--output", default="h1_motion.npz")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--xml", default="/workspace/mujoco_menagerie/unitree_h1/h1.xml")
    args = ap.parse_args()

    h1 = H1(args.xml)

    # Load AMASS
    raw = np.load(args.file, allow_pickle=True)
    poses, trans = raw['poses'][:, :72], raw['trans']
    src_fps = float(raw['mocap_framerate'])
    step = max(1, int(src_fps / args.fps))
    poses, trans = poses[::step], trans[::step]
    fps = src_fps / step
    dt = 1.0 / fps
    N = poses.shape[0]
    print(f"Data: {N} frames @ {fps:.0f} fps ({N*dt:.2f}s)")

    # SMPL FK → H1 coordinates
    print("SMPL FK...")
    smpl_h1 = np.zeros((N, 24, 3))
    for f in range(N):
        smpl_h1[f] = smpl_to_h1_coords(smpl_fk(poses[f], trans[f]))

    # Scale SMPL skeleton to match H1 proportions
    # Use T-pose leg lengths (from bone offsets, not from dynamic pose)
    smpl_tpose_leg = abs(SMPL_OFFSETS[1][1]) + abs(SMPL_OFFSETS[4][1]) + abs(SMPL_OFFSETS[7][1])
    # = 0.087 + 0.400 + 0.420 = 0.907m

    h1_zero_xpos, _ = h1.fk(np.zeros(h1.ndof), [0, 0, 1.06], [1, 0, 0, 0])
    h1_pelvis_z = h1_zero_xpos[1, 2]   # pelvis body (idx 1, after world)
    h1_ankle_z = h1_zero_xpos[6, 2]    # left ankle
    h1_leg = h1_pelvis_z - h1_ankle_z  # ~0.974m

    scale = h1_leg / smpl_tpose_leg
    print(f"Scale: SMPL T-pose leg={smpl_tpose_leg:.3f}m, H1 leg={h1_leg:.3f}m, scale={scale:.3f}")

    # Scale positions relative to root (pelvis) to match H1 proportions
    root_smpl = smpl_h1[:, 0:1, :].copy()
    smpl_h1 = (smpl_h1 - root_smpl) * scale + root_smpl

    # Root positions: set frame 0 pelvis at H1 standing height,
    # preserve dynamic height variations from SMPL motion
    root_positions = smpl_h1[:, 0, :].copy()
    z_offset = h1_pelvis_z - root_positions[0, 2]
    root_positions[:, 2] += z_offset
    smpl_h1[:, :, 2] += z_offset

    print(f"Root height: [{root_positions[:,2].min():.3f}, {root_positions[:,2].max():.3f}]")

    # Root orientations
    root_quats = np.zeros((N, 4))
    for f in range(N):
        root_quats[f] = smpl_root_to_h1(poses[f, :3])

    # Optimize DOFs
    print(f"Fitting {N} frames...")
    dofs = np.zeros((N, h1.ndof))
    prev = np.zeros(h1.ndof)
    for f in range(N):
        dofs[f], loss = fit_frame(h1, smpl_h1[f], root_positions[f], root_quats[f], prev)
        prev = dofs[f].copy()
        if (f+1) % 10 == 0: print(f"  {f+1}/{N}  loss={loss*1000:.3f}")

    dofs = dofs.astype(np.float32)

    # Final FK
    print("Final FK...")
    nb = h1.m.nbody - 1  # exclude world
    bpos = np.zeros((N, nb, 3), dtype=np.float32)
    brot = np.zeros((N, nb, 4), dtype=np.float32)
    for f in range(N):
        xp, xq = h1.fk(dofs[f], root_positions[f], root_quats[f])
        bpos[f] = xp[1:]  # skip world
        brot[f] = xq[1:]

    bnames = h1.body_names[1:]

    # Velocities
    dv = np.zeros_like(dofs); dv[1:] = (dofs[1:] - dofs[:-1]) / dt; dv[0] = dv[1]
    blv = np.zeros_like(bpos); blv[1:] = (bpos[1:] - bpos[:-1]) / dt; blv[0] = blv[1]
    bav = np.zeros((N, nb, 3), dtype=np.float32)
    for f in range(1, N):
        for b in range(nb):
            q0, q1 = brot[f-1,b], brot[f,b]
            r0 = sRot.from_quat([q0[1],q0[2],q0[3],q0[0]])
            r1 = sRot.from_quat([q1[1],q1[2],q1[3],q1[0]])
            bav[f,b] = r0.apply((r0.inv()*r1).as_rotvec()) / dt
    bav[0] = bav[1]

    # Report
    print(f"\n=== Result ===")
    print(f"Pelvis Z: [{bpos[:,0,2].min():.3f}, {bpos[:,0,2].max():.3f}]")
    ai = bnames.index("left_ankle_link")
    print(f"L ankle Z: [{bpos[:,ai,2].min():.3f}, {bpos[:,ai,2].max():.3f}]")
    ai = bnames.index("right_ankle_link")
    print(f"R ankle Z: [{bpos[:,ai,2].min():.3f}, {bpos[:,ai,2].max():.3f}]")
    for i, n in enumerate(h1.dof_names):
        print(f"  {n:25s}: [{dofs[:,i].min():7.3f}, {dofs[:,i].max():7.3f}]  lim=[{h1.lo[i]:.2f},{h1.hi[i]:.2f}]")

    np.savez(args.output, fps=int(fps), dof_names=np.array(h1.dof_names),
             body_names=np.array(bnames), dof_positions=dofs, dof_velocities=dv,
             body_positions=bpos, body_rotations=brot,
             body_linear_velocities=blv, body_angular_velocities=bav)
    print(f"\nSaved: {args.output}")

if __name__ == "__main__":
    main()
