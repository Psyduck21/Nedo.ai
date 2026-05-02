import os
import sys
import glob
import random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

# ─── Paths ────────────────────────────────────────────────────────────────────
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)
from models.skeleton_encoder import SkeletonEncoder
from scripts.dataset import compute_kinetic_features

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(base_dir, "models", "ssl_encoder_v5_final.pth")
OUTPUT_PATH = os.path.join(base_dir, "reconstruction_v5_kinetic.mp4")
FPS         = 30
DURATION_S  = 8
N_FRAMES    = FPS * DURATION_S   # 240 frames
TRAIL_LEN   = 5                  # number of ghost trail frames

# ─── MediaPipe Hand Skeleton Connections (local indices, 0-20 per hand) ───────
# Each hand has 21 landmarks. Connections define the "bones".
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm arch
    (5, 9), (9, 13), (13, 17),
]

# Global indices in the 543-landmark array
LEFT_HAND_START  = 501   # indices 501-521
RIGHT_HAND_START = 522   # indices 522-542
LEFT_WRIST_IDX   = LEFT_HAND_START   + 0   # 501
RIGHT_WRIST_IDX  = RIGHT_HAND_START  + 0   # 522


def draw_hand_skeleton(ax, landmarks_xy, color, lw=1.2, alpha=1.0, offset=0):
    """
    Draw MediaPipe hand bones as lines and return the segments.
    landmarks_xy: (42, 2) array — left hand first 21, right hand last 21.
    offset: 0 for left hand, 21 for right hand.
    """
    segs = []
    for (a, b) in HAND_CONNECTIONS:
        pa = landmarks_xy[offset + a]
        pb = landmarks_xy[offset + b]
        segs.append([pa, pb])
    if segs:
        lc = LineCollection(segs, colors=color, linewidths=lw, alpha=alpha, zorder=3)
        ax.add_collection(lc)


def main():
    print("=" * 60)
    print("   RECONSTRUCTION VISUALIZATION v5 — The Kinetic Engine")
    print("   (9-channel: pos+vel+acc | 100% Hand Blindfold | 16 Attn Heads)")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return
    model = SkeletonEncoder(in_channels=9, nhead=16).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Device: {device}")

    # ── Pick a real sequence ───────────────────────────────────────
    seq_dir   = os.path.join(base_dir, "Dataset", "processed_sequences")
    seq_files = glob.glob(os.path.join(seq_dir, "*.npy"))
    if not seq_files:
        print(f"ERROR: No sequences found in {seq_dir}")
        return

    valid_files = [f for f in seq_files
                   if np.load(f, mmap_mode='r').shape[0] >= N_FRAMES]
    if not valid_files:
        print(f"ERROR: No sequences with >= {N_FRAMES} frames.")
        return

    selected = random.choice(valid_files)
    print(f"Sequence: {os.path.basename(selected)}")

    full_data = np.load(selected)
    start     = random.randint(0, full_data.shape[0] - N_FRAMES)
    data      = full_data[start:start + N_FRAMES].copy()  # (240, 543, 3)

    # Build 9-channel kinetic features from raw position data
    # Apply total blindfold: zero all 9 channels of hands in every frame
    kinetic_full   = compute_kinetic_features(data)      # (240, 543, 9)
    kinetic_masked = kinetic_full.copy()
    kinetic_masked[:, 501:543, :] = 0.0                  # 100% blindfold

    batched = kinetic_masked.reshape(N_FRAMES // 60, 60, 543, 9)
    tensor  = torch.tensor(batched, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon_9d = model(tensor).cpu().numpy().reshape(N_FRAMES, 543, 9)

    # Extract position channels (0:3) for visualization — (240, 543, 3)
    recon = recon_9d[:, :, 0:3]

    print("Inference complete.")

    # ── Stable axis limits from actual data ───────────────────────
    all_xy = np.concatenate([data[:, :, :2], recon[:, :, :2]], axis=0)
    pad    = 0.15
    xmin, xmax = all_xy[:, :, 0].min() - pad, all_xy[:, :, 0].max() + pad
    ymin, ymax = -all_xy[:, :, 1].max() - pad, -all_xy[:, :, 1].min() + pad

    # ── Helper to get (x, y) for a landmark in screen coords ──────
    def xy(arr_frame, idx):
        return arr_frame[idx, 0], -arr_frame[idx, 1]

    def hand_xy(arr_frame, start_idx):
        """Returns (42, 2) array of (x, -y) for both hands."""
        pts = arr_frame[start_idx:start_idx + 42, :2].copy()
        pts[:, 1] = -pts[:, 1]
        return pts

    # ─── Figure: side-by-side ─────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 7), facecolor='#0d0d0d'
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.87, bottom=0.07, wspace=0.12)

    for ax in (ax_left, ax_right):
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='#555555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#333333')
            spine.set_linewidth(0.5)

    ax_left.set_title("Ground Truth  (Green = True Hands)",
                       color='white', fontsize=11, pad=8)
    ax_right.set_title("Model Reconstruction  (Red Bones + Wrist Trails)",
                        color='white', fontsize=11, pad=8)

    fig.suptitle(
        "Reconstruction v5: The Kinetic Engine — pos+vel+acc Hallucination",
        color='white', fontsize=13, fontweight='bold', y=0.97
    )

    frame_label = fig.text(0.5, 0.01, '', ha='center',
                            color='#888888', fontsize=9)

    # Legend for right panel
    legend_elems = [
        Line2D([0], [0], color='#ff3333', lw=2,         label='Reconstructed Bones'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#ff3333', markersize=9,
               linestyle='None',                         label='Wrists (Recon)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#ff3333', markersize=5,
               alpha=0.25, linestyle='None',             label='Wrist Trail'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#555555', markersize=5,
               alpha=0.4, linestyle='None',              label='Body / Face'),
    ]
    ax_right.legend(handles=legend_elems, loc='upper right',
                    facecolor='#1a1a1a', edgecolor='#444',
                    labelcolor='white', fontsize=8, framealpha=0.85)

    def update(frame):
        ax_left.cla()
        ax_right.cla()

        for ax in (ax_left, ax_right):
            ax.set_facecolor('#0d0d0d')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.tick_params(colors='#555555', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#333333')
                spine.set_linewidth(0.5)

        ax_left.set_title("Ground Truth  (Green = True Hands)",
                           color='white', fontsize=11, pad=8)
        ax_right.set_title("Model Reconstruction  (Red Bones + Wrist Trails)",
                            color='white', fontsize=11, pad=8)

        d = data[frame]
        r = recon[frame]

        # ── LEFT PANEL: Ground Truth ───────────────────────────────
        # Body / face
        bx = d[:501, 0]; by = -d[:501, 1]
        ax_left.scatter(bx, by, c='#444444', s=4, alpha=0.4, zorder=1)

        # True hand bones — left hand
        true_pts = hand_xy(d, 501)
        draw_hand_skeleton(ax_left, true_pts, color='#00ff88', lw=1.5, offset=0)
        draw_hand_skeleton(ax_left, true_pts, color='#00ff88', lw=1.5, offset=21)
        ax_left.scatter(true_pts[:, 0], true_pts[:, 1],
                        c='#00ff88', s=12, alpha=0.7, zorder=4)

        # ── RIGHT PANEL: Reconstruction ────────────────────────────
        # Body / face (from reconstructed output)
        rx_b = r[:501, 0]; ry_b = -r[:501, 1]
        ax_right.scatter(rx_b, ry_b, c='#444444', s=4, alpha=0.4, zorder=1)

        # Ghost overlay of true hands (faint green)
        ax_right.scatter(true_pts[:, 0], true_pts[:, 1],
                         c='#00ff88', s=12, alpha=0.18, zorder=2)

        # Reconstructed hand bones
        rec_pts = hand_xy(r, 501)
        draw_hand_skeleton(ax_right, rec_pts, color='#ff3333', lw=1.5, alpha=0.9, offset=0)
        draw_hand_skeleton(ax_right, rec_pts, color='#ff3333', lw=1.5, alpha=0.9, offset=21)

        # Only plot wrist dots (landmark 0 of each hand), NOT all 42 joints
        for wrist_local in [0, 21]:
            wx, wy = rec_pts[wrist_local, 0], rec_pts[wrist_local, 1]
            ax_right.scatter(wx, wy, c='#ff3333', s=90, zorder=5,
                             edgecolors='#ffaaaa', linewidths=0.8)

        # Motion trails for wrists (TRAIL_LEN previous frames, fading alpha)
        for t in range(1, TRAIL_LEN + 1):
            trail_frame = max(0, frame - t)
            trail_alpha = (TRAIL_LEN - t + 1) / (TRAIL_LEN * 5)  # 0.04 → 0.20
            tr = recon[trail_frame]
            trail_pts = hand_xy(tr, 501)
            for wrist_local in [0, 21]:
                wx, wy = trail_pts[wrist_local, 0], trail_pts[wrist_local, 1]
                ax_right.scatter(wx, wy, c='#ff3333',
                                 s=50 - t * 6,
                                 alpha=trail_alpha + 0.05 * (TRAIL_LEN - t),
                                 zorder=4, edgecolors='none')

        ax_right.legend(handles=legend_elems, loc='upper right',
                        facecolor='#1a1a1a', edgecolor='#444',
                        labelcolor='white', fontsize=8, framealpha=0.85)

        frame_label.set_text(
            f"Frame {frame + 1:03d}/{N_FRAMES}   t = {frame / FPS:.2f}s"
        )

    # ── Render ────────────────────────────────────────────────────
    ani = animation.FuncAnimation(
        fig, update, frames=N_FRAMES, interval=1000 // FPS, blit=False
    )

    print(f"\nRendering {N_FRAMES} frames @ {FPS}fps ({DURATION_S}s) ...")
    try:
        writer = animation.FFMpegWriter(
            fps=FPS, bitrate=3000,
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
        )
        ani.save(OUTPUT_PATH, writer=writer, dpi=120)
        print(f"Saved: {OUTPUT_PATH}")
    except Exception as e:
        fallback = OUTPUT_PATH.replace('.mp4', '.gif')
        print(f"FFmpeg failed ({e}). Falling back to GIF...")
        ani.save(fallback, writer='pillow', fps=FPS)
        print(f"Saved (GIF): {fallback}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
