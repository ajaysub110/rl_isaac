#!/usr/bin/env python3
"""Visualize retargeted H1 motion from MotionLoader-format NPZ as animated GIF.

Reads the body_positions from the NPZ and renders the H1 skeleton with bones.

Usage:
    python visualize_retarget.py --file outputs/h1_roundhouse.npz --output outputs/h1_retarget.gif
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# H1 skeleton bone connectivity (link_index pairs)
H1_BONES = [
    # Left leg
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    # Right leg
    (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    # Torso
    (0, 11),
    # Left arm
    (11, 12), (12, 13), (13, 14), (14, 15),
    # Right arm
    (11, 16), (16, 17), (17, 18), (18, 19),
]

# Bone colors by body part
def get_bone_color(i, j):
    if j <= 5:  # left leg
        return "#2196F3"  # blue
    elif j <= 10:  # right leg
        return "#F44336"  # red
    elif j == 11:  # torso
        return "#4CAF50"  # green
    elif j <= 15:  # left arm
        return "#03A9F4"  # light blue
    else:  # right arm
        return "#FF5722"  # deep orange


def create_animation(body_positions, body_names, fps, output_path):
    """Create animated GIF from body positions.

    Args:
        body_positions: (N, B, 3) body positions
        body_names: list of body names
        fps: framerate
        output_path: path for the GIF
    """
    N, B, _ = body_positions.shape

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#1a1a2e')

    # Scene bounds
    all_min = body_positions.reshape(-1, 3).min(axis=0)
    all_max = body_positions.reshape(-1, 3).max(axis=0)
    center = 0.5 * (all_max + all_min)
    extent = 0.65 * (all_max - all_min).max()

    # Ensure minimum extent and a good view
    extent = max(extent, 0.8)

    def update(frame):
        ax.clear()
        ax.set_facecolor('#1a1a2e')

        verts = body_positions[frame]

        # Draw bones
        for (i, j) in H1_BONES:
            if i < B and j < B:
                xs = [verts[i, 0], verts[j, 0]]
                ys = [verts[i, 1], verts[j, 1]]
                zs = [verts[i, 2], verts[j, 2]]
                ax.plot(xs, ys, zs, color=get_bone_color(i, j), linewidth=3, alpha=0.9)

        # Draw joints
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                   c='white', s=25, depthshade=False, zorder=5)

        # Mark pelvis
        ax.scatter(*verts[0], c='yellow', s=60, depthshade=False, zorder=6, marker='*')

        # Axes
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(max(0, center[2] - extent), center[2] + extent)

        # Ground plane
        gx, gy = np.meshgrid(
            [center[0] - extent, center[0] + extent],
            [center[1] - extent, center[1] + extent]
        )
        ax.plot_surface(gx, gy, np.zeros_like(gx), color='#2d2d44', alpha=0.3)

        ax.set_xlabel('X (fwd)', color='white', fontsize=8)
        ax.set_ylabel('Y (left)', color='white', fontsize=8)
        ax.set_zlabel('Z (up)', color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=6)

        t = frame / fps
        ax.set_title(f'H1 Retargeted Motion — Frame {frame}/{N}  ({t:.2f}s)',
                     color='white', fontsize=11, fontweight='bold')

    print(f"Rendering {N} frames at {fps} fps...")
    anim = animation.FuncAnimation(fig, update, frames=N, interval=1000/fps, blit=False)
    anim.save(output_path, writer='pillow', fps=min(fps, 15))  # cap GIF fps to keep file small
    plt.close(fig)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="H1 motion NPZ file")
    parser.add_argument("--output", type=str, default="h1_retarget.gif", help="Output GIF")
    args = parser.parse_args()

    data = np.load(args.file, allow_pickle=True)
    body_positions = data['body_positions']   # (N, B, 3)
    body_names = data['body_names'].tolist()
    fps = int(data['fps'])
    dof_positions = data['dof_positions']     # (N, D)
    dof_names = data['dof_names'].tolist()

    print(f"Loaded: {args.file}")
    print(f"  Frames: {body_positions.shape[0]}, FPS: {fps}")
    print(f"  Bodies: {len(body_names)}")
    print(f"  DOFs: {len(dof_names)}")

    # Quick sanity check
    print(f"\n  Pelvis height range: [{body_positions[:, 0, 2].min():.3f}, {body_positions[:, 0, 2].max():.3f}]")

    # Print end-effector positions (ankles and elbows)
    for name, idx in [("left_ankle_link", 5), ("right_ankle_link", 10),
                      ("left_elbow_link", 15), ("right_elbow_link", 19)]:
        if idx < body_positions.shape[1]:
            z_range = body_positions[:, idx, 2]
            print(f"  {name} Z: [{z_range.min():.3f}, {z_range.max():.3f}]")

    create_animation(body_positions, body_names, fps, args.output)


if __name__ == "__main__":
    main()
