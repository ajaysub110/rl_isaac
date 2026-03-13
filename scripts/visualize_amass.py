#!/usr/bin/env python3
"""Visualize AMASS motion capture data (SMPL format) as an animated 3D skeleton GIF.

Uses direct forward kinematics on the SMPL kinematic tree — no SMPL model files needed.
Only requires the poses (axis-angle) and trans (root translation) from the AMASS NPZ.

Usage:
    python visualize_amass.py --file /path/to/amass.npz --output motion.gif
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ── SMPL 24-joint kinematic tree ──────────────────────────────────────────────
# Joint names for the 24 SMPL body joints
SMPL_JOINT_NAMES = [
    "Pelvis",       # 0
    "L_Hip",        # 1
    "R_Hip",        # 2
    "Spine1",       # 3
    "L_Knee",       # 4
    "R_Knee",       # 5
    "Spine2",       # 6
    "L_Ankle",      # 7
    "R_Ankle",      # 8
    "Spine3",       # 9
    "L_Foot",       # 10
    "R_Foot",       # 11
    "Neck",         # 12
    "L_Collar",     # 13
    "R_Collar",     # 14
    "Head",         # 15
    "L_Shoulder",   # 16
    "R_Shoulder",   # 17
    "L_Elbow",      # 18
    "R_Elbow",      # 19
    "L_Wrist",      # 20
    "R_Wrist",      # 21
    "L_Hand",       # 22
    "R_Hand",       # 23
]

# Parent indices (-1 = root)
SMPL_PARENTS = [
    -1,  # 0  Pelvis
     0,  # 1  L_Hip
     0,  # 2  R_Hip
     0,  # 3  Spine1
     1,  # 4  L_Knee
     2,  # 5  R_Knee
     3,  # 6  Spine2
     4,  # 7  L_Ankle
     5,  # 8  R_Ankle
     6,  # 9  Spine3
     7,  # 10 L_Foot
     8,  # 11 R_Foot
     9,  # 12 Neck
     9,  # 13 L_Collar
     9,  # 14 R_Collar
    12,  # 15 Head
    13,  # 16 L_Shoulder
    14,  # 17 R_Shoulder
    16,  # 18 L_Elbow
    17,  # 19 R_Elbow
    18,  # 20 L_Wrist
    19,  # 21 R_Wrist
    20,  # 22 L_Hand
    21,  # 23 R_Hand
]

# Bones to draw (pairs of joint indices)
SMPL_BONES = [
    (0, 1), (0, 2), (0, 3),      # pelvis → hips, spine
    (1, 4), (2, 5),               # hips → knees
    (4, 7), (5, 8),               # knees → ankles
    (7, 10), (8, 11),             # ankles → feet
    (3, 6), (6, 9),               # spine chain
    (9, 12), (9, 13), (9, 14),    # spine3 → neck, collars
    (12, 15),                      # neck → head
    (13, 16), (14, 17),           # collars → shoulders
    (16, 18), (17, 19),           # shoulders → elbows
    (18, 20), (19, 21),           # elbows → wrists
    (20, 22), (21, 23),           # wrists → hands
]

# Approximate bone rest-lengths in SMPL T-pose (meters) — used for FK offsets
# These are approximate average values for a male SMPL model
SMPL_T_POSE_OFFSETS = np.zeros((24, 3))
SMPL_T_POSE_OFFSETS[1]  = [ 0.065, -0.087, -0.030]   # L_Hip
SMPL_T_POSE_OFFSETS[2]  = [-0.065, -0.087, -0.030]   # R_Hip
SMPL_T_POSE_OFFSETS[3]  = [ 0.000,  0.120,  0.010]   # Spine1
SMPL_T_POSE_OFFSETS[4]  = [ 0.000, -0.400,  0.000]   # L_Knee
SMPL_T_POSE_OFFSETS[5]  = [ 0.000, -0.400,  0.000]   # R_Knee
SMPL_T_POSE_OFFSETS[6]  = [ 0.000,  0.130,  0.000]   # Spine2
SMPL_T_POSE_OFFSETS[7]  = [ 0.000, -0.420,  0.000]   # L_Ankle
SMPL_T_POSE_OFFSETS[8]  = [ 0.000, -0.420,  0.000]   # R_Ankle
SMPL_T_POSE_OFFSETS[9]  = [ 0.000,  0.180,  0.000]   # Spine3
SMPL_T_POSE_OFFSETS[10] = [ 0.000, -0.060,  0.120]   # L_Foot
SMPL_T_POSE_OFFSETS[11] = [ 0.000, -0.060,  0.120]   # R_Foot
SMPL_T_POSE_OFFSETS[12] = [ 0.000,  0.130,  0.020]   # Neck
SMPL_T_POSE_OFFSETS[13] = [ 0.070,  0.060, -0.010]   # L_Collar
SMPL_T_POSE_OFFSETS[14] = [-0.070,  0.060, -0.010]   # R_Collar
SMPL_T_POSE_OFFSETS[15] = [ 0.000,  0.120,  0.020]   # Head
SMPL_T_POSE_OFFSETS[16] = [ 0.120,  0.000,  0.000]   # L_Shoulder
SMPL_T_POSE_OFFSETS[17] = [-0.120,  0.000,  0.000]   # R_Shoulder
SMPL_T_POSE_OFFSETS[18] = [ 0.260,  0.000,  0.000]   # L_Elbow
SMPL_T_POSE_OFFSETS[19] = [-0.260,  0.000,  0.000]   # R_Elbow
SMPL_T_POSE_OFFSETS[20] = [ 0.260,  0.000,  0.000]   # L_Wrist
SMPL_T_POSE_OFFSETS[21] = [-0.260,  0.000,  0.000]   # R_Wrist
SMPL_T_POSE_OFFSETS[22] = [ 0.100,  0.000,  0.000]   # L_Hand
SMPL_T_POSE_OFFSETS[23] = [-0.100,  0.000,  0.000]   # R_Hand


def axis_angle_to_rotation_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to 3x3 rotation matrix via Rodrigues' formula."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def forward_kinematics(poses: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Compute 3D joint positions from SMPL poses and root translation.

    Args:
        poses: (N, 72+) axis-angle rotations for 24 body joints
        trans: (N, 3) root translation

    Returns:
        positions: (N, 24, 3) joint positions in world frame
    """
    num_frames = poses.shape[0]
    positions = np.zeros((num_frames, 24, 3))

    for f in range(num_frames):
        # Per-joint rotation matrices
        local_rots = np.zeros((24, 3, 3))
        for j in range(24):
            aa = poses[f, j*3:(j+1)*3]
            local_rots[j] = axis_angle_to_rotation_matrix(aa)

        # Forward kinematics: accumulate transforms
        global_rots = np.zeros((24, 3, 3))
        global_pos = np.zeros((24, 3))

        # Root
        global_rots[0] = local_rots[0]
        global_pos[0] = trans[f]

        for j in range(1, 24):
            parent = SMPL_PARENTS[j]
            # This child's position = parent_pos + parent_rot * offset
            global_rots[j] = global_rots[parent] @ local_rots[j]
            global_pos[j] = global_pos[parent] + global_rots[parent] @ SMPL_T_POSE_OFFSETS[j]

        positions[f] = global_pos

    return positions


def create_animation(positions: np.ndarray, fps: float, output_path: str,
                     subsample: int = 4, render_scene: bool = False):
    """Create and save an animated GIF of the skeleton.

    Args:
        positions: (N, 24, 3) joint positions
        fps: original framerate
        output_path: path to save the GIF
        subsample: take every Nth frame to reduce GIF size
        render_scene: if True, show full scene extent; else zoom to skeleton
    """
    positions = positions[::subsample]
    effective_fps = fps / subsample
    num_frames = positions.shape[0]

    # Color scheme for different body parts
    bone_colors = {}
    for b in SMPL_BONES:
        # Left side = blue, right side = red, spine = green, extremities = orange
        j = max(b)
        name = SMPL_JOINT_NAMES[j]
        if "L_" in name:
            bone_colors[b] = "#2196F3"  # blue
        elif "R_" in name:
            bone_colors[b] = "#F44336"  # red
        elif any(s in name for s in ["Spine", "Neck", "Head", "Pelvis"]):
            bone_colors[b] = "#4CAF50"  # green
        else:
            bone_colors[b] = "#FF9800"  # orange

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#1a1a2e')

    # Precompute scene bounds
    all_min = positions.reshape(-1, 3).min(axis=0)
    all_max = positions.reshape(-1, 3).max(axis=0)
    all_center = 0.5 * (all_max + all_min)
    all_extent = 0.6 * (all_max - all_min).max()

    def update(frame):
        ax.clear()
        ax.set_facecolor('#1a1a2e')

        verts = positions[frame]

        # Draw bones
        for bone in SMPL_BONES:
            xs = [verts[bone[0], 0], verts[bone[1], 0]]
            ys = [verts[bone[0], 1], verts[bone[1], 1]]
            zs = [verts[bone[0], 2], verts[bone[1], 2]]
            color = bone_colors[bone]
            ax.plot(xs, ys, zs, color=color, linewidth=2.5, alpha=0.9)

        # Draw joints
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                   c='white', s=20, depthshade=False, zorder=5, alpha=0.9)

        # Scene bounds
        if render_scene:
            cx, cy, cz = all_center
            ext = all_extent
        else:
            v_min = verts.min(axis=0)
            v_max = verts.max(axis=0)
            cx, cy, cz = 0.5 * (v_max + v_min)
            ext = 0.6 * max((v_max - v_min).max(), 0.5)

        ax.set_xlim(cx - ext, cx + ext)
        ax.set_ylim(cy - ext, cy + ext)
        ax.set_zlim(cz - ext, cz + ext)

        # Ground plane
        gx, gy = np.meshgrid([cx - ext, cx + ext], [cy - ext, cy + ext])
        ax.plot_surface(gx, gy, np.zeros_like(gx), color='#2d2d44', alpha=0.3)

        ax.set_xlabel('X', color='white', fontsize=8)
        ax.set_ylabel('Y', color='white', fontsize=8)
        ax.set_zlabel('Z', color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=6)
        time_s = frame * subsample / fps
        ax.set_title(f'AMASS Motion — Frame {frame * subsample}/{len(positions) * subsample}  '
                     f'({time_s:.2f}s / {len(positions) * subsample / fps:.2f}s)',
                     color='white', fontsize=11, fontweight='bold')

    print(f"Rendering {num_frames} frames at {effective_fps:.1f} fps (subsampled {subsample}x)...")
    anim = animation.FuncAnimation(fig, update, frames=num_frames,
                                   interval=1000 / effective_fps, blit=False)
    anim.save(output_path, writer='pillow', fps=effective_fps)
    plt.close(fig)
    print(f"Saved animation to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize AMASS motion capture data")
    parser.add_argument("--file", type=str, required=True, help="AMASS NPZ file path")
    parser.add_argument("--output", type=str, default="amass_motion.gif", help="Output GIF path")
    parser.add_argument("--subsample", type=int, default=4, help="Take every Nth frame")
    parser.add_argument("--scene", action="store_true", help="Render full scene extent")
    args = parser.parse_args()

    # Load AMASS data
    data = np.load(args.file, allow_pickle=True)
    poses = data['poses']       # (N, 156) for SMPL+H, we use first 72
    trans = data['trans']       # (N, 3)
    fps = float(data['mocap_framerate'])

    print(f"Loaded: {args.file}")
    print(f"  Frames:    {poses.shape[0]}")
    print(f"  FPS:       {fps}")
    print(f"  Duration:  {poses.shape[0] / fps:.2f}s")
    print(f"  Gender:    {data['gender']}")
    print(f"  Pose dims: {poses.shape[1]} ({'SMPL+H' if poses.shape[1] > 72 else 'SMPL'})")

    # Forward kinematics (only use first 72 params = 24 body joints)
    positions = forward_kinematics(poses[:, :72], trans)

    print(f"\nFK results:")
    print(f"  Joint positions shape: {positions.shape}")
    print(f"  Height range (Z):  [{positions[:,:,2].min():.3f}, {positions[:,:,2].max():.3f}]")
    print(f"  X range:           [{positions[:,:,0].min():.3f}, {positions[:,:,0].max():.3f}]")
    print(f"  Y range:           [{positions[:,:,1].min():.3f}, {positions[:,:,1].max():.3f}]")

    # Create animation
    create_animation(positions, fps, args.output, subsample=args.subsample, render_scene=args.scene)


if __name__ == "__main__":
    main()
