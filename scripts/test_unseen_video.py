"""
test_unseen_video.py — Nedo.ai Out-of-Distribution Inference Pipeline
======================================================================
Full pipeline: raw video → MediaPipe landmarks → v5 9-channel kinetic
features → SSL encoder inference → side-by-side visualization with MSE log.

Usage:
    python scripts/test_unseen_video.py --video path/to/video.mp4
    python scripts/test_unseen_video.py --video path/to/video.mov --output my_output.mp4
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# Suppress MediaPipe / TensorFlow C++ noise
os.environ['GLOG_minloglevel']     = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_MIN_LEVEL']   = '3'

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from contextlib import contextmanager
from tqdm import tqdm

import mediapipe as mp
BaseOptions           = mp.tasks.BaseOptions
HolisticLandmarker   = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)

from models.skeleton_encoder import SkeletonEncoder
from scripts.dataset import compute_kinetic_features

# ─── MediaPipe Hand Skeleton Connections (local 0-20 per hand) ────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

LEFT_HAND_OFFSET  = 501
RIGHT_HAND_OFFSET = 522
WINDOW_SIZE       = 60
STEP_SIZE         = 30   # 50% overlap for smooth video output


# ─── Helpers ──────────────────────────────────────────────────────────────────

@contextmanager
def suppress_stderr():
    fd      = sys.stderr.fileno()
    def_pad = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(def_pad, fd)
        os.close(devnull)
        os.close(def_pad)


def get_landmarks_array(landmark_list, num_expected):
    if not landmark_list or len(landmark_list) == 0:
        return np.zeros((num_expected, 3), dtype=np.float32)
    lms = landmark_list[0] if isinstance(landmark_list[0], list) else landmark_list
    arr = np.zeros((num_expected, 3), dtype=np.float32)
    for i, lm in enumerate(lms):
        if i >= num_expected:
            break
        arr[i] = [lm.x, lm.y, lm.z]
    return arr


def hip_centric_normalize(all_landmarks, pose_lms, pose_world_lms):
    """
    Exact same normalization as NedoAi/extract_landmarks.py:
      - Subtract mid-hip (screen coords)
      - Divide by shoulder width (world coords) for scale invariance
    """
    if (pose_lms is None or len(pose_lms) < 25 or
            pose_world_lms is None or len(pose_world_lms) < 25):
        return all_landmarks

    lms       = pose_lms[0]  if isinstance(pose_lms[0],       list) else pose_lms
    world_lms = pose_world_lms[0] if isinstance(pose_world_lms[0], list) else pose_world_lms

    if len(lms) < 25 or len(world_lms) < 25:
        return all_landmarks

    left_hip  = np.array([lms[23].x, lms[23].y, lms[23].z])
    right_hip = np.array([lms[24].x, lms[24].y, lms[24].z])
    mid_hip   = (left_hip + right_hip) / 2.0

    w_left_sh  = np.array([world_lms[11].x, world_lms[11].y, world_lms[11].z])
    w_right_sh = np.array([world_lms[12].x, world_lms[12].y, world_lms[12].z])
    shoulder_w = np.linalg.norm(w_left_sh - w_right_sh)

    if shoulder_w > 0:
        non_zero = np.any(all_landmarks != 0, axis=1)
        all_landmarks[non_zero] = (all_landmarks[non_zero] - mid_hip) / shoulder_w

    return all_landmarks


def draw_hand_bones(frame, pts_uv, color, thickness=1):
    """Standard bone drawing with clipping."""
    H, W = frame.shape[:2]
    for (a, b) in HAND_CONNECTIONS:
        p1, p2 = (int(pts_uv[a][0]), int(pts_uv[a][1])), (int(pts_uv[b][0]), int(pts_uv[b][1]))
        ret, cp1, cp2 = cv2.clipLine((0, 0, W, H), p1, p2)
        if ret:
            cv2.line(frame, cp1, cp2, color, thickness, cv2.LINE_AA)

def draw_neon_bones(frame, pts_uv, base_color=(0, 0, 255)):
    """Vivid neon effect: Thick red glow with white core."""
    H, W = frame.shape[:2]
    for (a, b) in HAND_CONNECTIONS:
        p1, p2 = (int(pts_uv[a][0]), int(pts_uv[a][1])), (int(pts_uv[b][0]), int(pts_uv[b][1]))
        ret, cp1, cp2 = cv2.clipLine((0, 0, W, H), p1, p2)
        if ret:
            # Red Glow
            cv2.line(frame, cp1, cp2, base_color, 4, cv2.LINE_AA)
            # White Core
            cv2.line(frame, cp1, cp2, (255, 255, 255), 1, cv2.LINE_AA)


def norm_to_px(xy_norm, W, H):
    """Convert normalized (x,y) to pixel coords."""
    return np.column_stack([xy_norm[:, 0] * W, xy_norm[:, 1] * H])


# ─── Stage 1: Extract Landmarks ───────────────────────────────────────────────

def extract_landmarks(video_path: str, holistic_model_path: str):
    """
    Run MediaPipe HolisticLandmarker frame-by-frame.
    Returns:
        frames     : list of BGR frames (original video)
        landmarks  : np.ndarray (T, 543, 3) hip-centric normalized
        fps        : float
    """
    print(f"[1/4] Extracting landmarks from: {os.path.basename(video_path)}")

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=holistic_model_path,
            delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=VisionRunningMode.VIDEO
    )

    with suppress_stderr():
        landmarker = HolisticLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames    = []
    seq       = []
    raw_seq   = []
    frame_idx = 0

    with tqdm(desc="  Extracting", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            ts_ms    = int(frame_idx * (1000.0 / fps))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with suppress_stderr():
                result = landmarker.detect_for_video(mp_image, ts_ms)

            face  = get_landmarks_array(result.face_landmarks,  468)
            pose  = get_landmarks_array(result.pose_landmarks,   33)
            lh    = get_landmarks_array(result.left_hand_landmarks,  21)
            rh    = get_landmarks_array(result.right_hand_landmarks, 21)
            raw   = np.concatenate([face, pose, lh, rh], axis=0)  # (543, 3)

            raw_seq.append(raw.copy())

            raw = hip_centric_normalize(
                raw,
                result.pose_landmarks,
                getattr(result, 'pose_world_landmarks', None)
            )
            seq.append(raw)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    landmarker.close()

    landmarks = np.array(seq, dtype=np.float32)   # (T, 543, 3)
    raw_landmarks = np.array(raw_seq, dtype=np.float32)
    print(f"  Extracted {len(frames)} frames @ {fps:.1f} fps  |  Landmarks: {landmarks.shape}")
    return frames, landmarks, raw_landmarks, fps


# ─── Stage 2: Build 9-Channel Kinetic Windows ─────────────────────────────────

def build_kinetic_windows(landmarks: np.ndarray):
    """
    Compute pos+vel+acc (9 channels), apply total blindfold on hands,
    and slice into sliding windows of 60 frames (step=30).
    Returns:
        windows_masked : (N, 60, 543, 9) — model input
        windows_full   : (N, 60, 543, 9) — ground truth (with real hands)
        window_starts  : list of start frame indices
    """
    print("[2/4] Building 9-channel kinetic windows...")

    kinetic = compute_kinetic_features(landmarks)  # (T, 543, 9)
    T = kinetic.shape[0]

    windows_masked = []
    windows_full   = []
    window_starts  = []

    for start in range(0, T - WINDOW_SIZE + 1, STEP_SIZE):
        win  = kinetic[start:start + WINDOW_SIZE].copy()   # (60, 543, 9)
        mwin = win.copy()
        mwin[:, 501:543, :] = 0.0    # Total blindfold on hands
        windows_full.append(win)
        windows_masked.append(mwin)
        window_starts.append(start)

    print(f"  {len(windows_masked)} windows  (size={WINDOW_SIZE}, step={STEP_SIZE})")
    return (np.array(windows_masked),
            np.array(windows_full),
            window_starts)


# ─── Stage 3: SSL Encoder Inference ───────────────────────────────────────────

def run_inference(windows_masked: np.ndarray, model_path: str, device: torch.device):
    """
    Loads ssl_encoder_v5 and runs inference on all windows.
    Returns:
        recon_windows : (N, 60, 543, 3) — reconstructed position only
    """
    print(f"[3/4] Running v5 Kinetic Encoder inference...")

    model = SkeletonEncoder(in_channels=9, nhead=16).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_path}")
    print(f"  Parameters: {n_params:,}  |  Device: {device}")

    recon_windows = []
    batch_size = 8

    with torch.no_grad():
        for i in range(0, len(windows_masked), batch_size):
            batch = torch.tensor(
                windows_masked[i:i + batch_size], dtype=torch.float32
            ).to(device)
            out = model(batch)  # (B, 60, 543, 9)
            # Extract position channels only (0:3)
            recon_windows.append(out.cpu().numpy()[:, :, :, 0:3])

    recon_windows = np.concatenate(recon_windows, axis=0)  # (N, 60, 543, 3)

    # ── MSE Report ────────────────────────────────────────────────────────────
    gt_pos = windows_masked[:, :, 501:543, 0:3]   # All zeros (blindfolded input)
    # Compare against ground truth from full windows (passed separately via CLI)
    print(f"  Inference complete. Output shape: {recon_windows.shape}")
    return recon_windows


def compute_reconstruction_mse(windows_full: np.ndarray, recon_windows: np.ndarray):
    """MSE between real hand positions and hallucinated hands."""
    gt_hands   = windows_full[:, :, 501:543, 0:3]    # (N, 60, 42, 3)
    pred_hands = recon_windows[:, :, 501:543, :]       # (N, 60, 42, 3)
    mse        = float(np.mean((gt_hands - pred_hands) ** 2))
    rmse       = float(np.sqrt(mse))
    print(f"\n{'─'*55}")
    print(f"  RECONSTRUCTION ERROR (unseen person):")
    print(f"    MSE  : {mse:.6f}")
    print(f"    RMSE : {rmse:.6f}  (in normalized shoulder-width units)")
    print(f"{'─'*55}\n")
    return mse, rmse


# ─── Stage 4: Side-by-Side Visualization ──────────────────────────────────────

def build_output_video(
    frames, landmarks, raw_landmarks, recon_windows, window_starts, fps, output_path
):
    """
    Produce side-by-side MP4:
      Left  — real video + MediaPipe skeleton overlay
      Right — black canvas + red hallucinated hand bones + wrist trails
    """
    print(f"[4/4] Rendering output video: {output_path}")

    H, W = frames[0].shape[:2]
    RIGHT_W = int(W * 1.5)
    OUT_W = W + RIGHT_W
    TRAIL = 8

    # Map each frame to its best reconstructed window
    T = len(frames)
    frame_recon = np.zeros((T, 543, 3), dtype=np.float32)
    frame_count = np.zeros(T, dtype=np.int32)

    for wi, start in enumerate(window_starts):
        for t in range(WINDOW_SIZE):
            fi = start + t
            if fi < T:
                frame_recon[fi] += recon_windows[wi, t]
                frame_count[fi] += 1

    # Average overlapping windows
    for fi in range(T):
        if frame_count[fi] > 0:
            frame_recon[fi] /= frame_count[fi]
    OUT_W = W + RIGHT_W
    TRAIL = 8

    # ... (skipping window mapping logic same as before) ...

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (OUT_W, H))

    wrist_trail = [] 

    for fi, frame in enumerate(tqdm(frames, desc="  Rendering", unit="frame")):
        left = frame.copy()
        raw_lm = raw_landmarks[fi]
        for v in range(501):
            x, y = int(raw_lm[v, 0] * W), int(raw_lm[v, 1] * H)
            if 0 < x < W and 0 < y < H:
                cv2.circle(left, (x, y), 2, (100, 100, 100), -1)
        for offset in [LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET]:
            pts = norm_to_px(raw_lm[offset:offset + 21, :2], W, H)
            draw_hand_bones(left, pts, (0, 220, 100), thickness=2)

        # ── RIGHT: Widescreen Stage ───────────────────────────────────────────
        right_canvas = np.zeros((H, RIGHT_W, 3), dtype=np.uint8) 
        
        # Draw Subtle Signing Box Grid
        for gx in range(0, RIGHT_W, 50):
            cv2.line(right_canvas, (gx, 0), (gx, H), (20, 20, 20), 1)
        for gy in range(0, H, 50):
            cv2.line(right_canvas, (0, gy), (RIGHT_W, gy), (20, 20, 20), 1)

        rec = frame_recon[fi]
        lm_norm = landmarks[fi]

        # WIDESCREEN SCALING
        DRAW_SCALE = H * 0.25 
        c_x, c_y = RIGHT_W // 2, H // 2

        def ghost_to_px(pts_norm):
            pts = np.zeros_like(pts_norm)
            pts[:, 0] = c_x + (pts_norm[:, 0] * DRAW_SCALE)
            pts[:, 1] = c_y + (pts_norm[:, 1] * DRAW_SCALE)
            return pts

        # Draw centered body skeleton
        for v in range(501):
            px, py = ghost_to_px(lm_norm[v:v+1, :2])[0]
            if 0 < px < RIGHT_W and 0 < py < H:
                cv2.circle(right_canvas, (int(px), int(py)), 2, (60, 60, 60), -1)

        # Neon Hallucinated Bones
        for offset_local, offset_global in [(0, LEFT_HAND_OFFSET), (21, RIGHT_HAND_OFFSET)]:
            rec_hand = rec[offset_global:offset_global + 21, :2]
            pts = ghost_to_px(rec_hand)
            draw_neon_bones(right_canvas, pts, (0, 0, 255))

        # Glowing Wrists
        for offset_global in [LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET]:
            pts_px = ghost_to_px(rec[offset_global:offset_global+1, :2])
            wx, wy = int(pts_px[0, 0]), int(pts_px[0, 1])
            if 0 < wx < RIGHT_W and 0 < wy < H:
                cv2.circle(right_canvas, (wx, wy), 10, (0, 0, 255), -1)
                cv2.circle(right_canvas, (wx, wy), 12, (180, 180, 255), 2)

        # Trails
        l_w = ghost_to_px(rec[LEFT_HAND_OFFSET:LEFT_HAND_OFFSET+1, :2])[0]
        r_w = ghost_to_px(rec[RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET+1, :2])[0]
        wrist_trail.append((int(l_w[0]), int(l_w[1]), int(r_w[0]), int(r_w[1])))
        if len(wrist_trail) > TRAIL: wrist_trail.pop(0)

        for ti, (lx, ly, rx, ry) in enumerate(reversed(wrist_trail[:-1])):
            alpha = int(255 * (ti + 1) / TRAIL * 0.6)
            overlay = right_canvas.copy()
            if 0 < lx < RIGHT_W and 0 < ly < H: cv2.circle(overlay, (lx, ly), 5, (255, 255, 255), -1)
            if 0 < rx < RIGHT_W and 0 < ry < H: cv2.circle(overlay, (rx, ry), 5, (255, 255, 255), -1)
            cv2.addWeighted(overlay, alpha / 255.0, right_canvas, 1 - alpha / 255.0, 0, right_canvas)

        cv2.putText(right_canvas, "v5 Widescreen Ghost", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Combine
        canvas = np.concatenate([left, right_canvas], axis=1)
        out.write(canvas)
        cv2.putText(
            canvas,
            f"v5 Kinetic Engine  |  9-Channel Inference  |  Frame {fi+1}/{T}",
            (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (220, 220, 220), 1, cv2.LINE_AA
        )

        out.write(canvas)

    out.release()
    print(f"  Saved: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nedo.ai OOD inference — v5 Kinetic Engine"
    )
    parser.add_argument("--video",  required=True,
                        help="Path to input .mp4 or .mov video")
    parser.add_argument("--output", default=None,
                        help="Output .mp4 path (default: <video_name>_v5_ghost.mp4)")
    parser.add_argument("--model",  default=None,
                        help="Path to ssl_encoder_v5 .pth (default: models/ssl_encoder_v5_final.pth)")
    parser.add_argument("--holistic", default=None,
                        help="Path to holistic.task (default: NedoAi/models/holistic.task)")
    args = parser.parse_args()

    video_path    = os.path.abspath(args.video)   # resolve relative to shell CWD
    model_path    = args.model    or os.path.join(base_dir, "models", "ssl_encoder_v5_final.pth")
    holistic_path = args.holistic or os.path.join(base_dir, "NedoAi", "models", "holistic.task")

    stem       = os.path.splitext(os.path.basename(video_path))[0]
    output_path = args.output or os.path.join(base_dir, "results", f"{stem}_v5_ghost.mp4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Nedo.ai — v5 Kinetic Engine OOD Inference")
    print("=" * 60)
    print(f"  Video   : {video_path}")
    print(f"  Model   : {model_path}")
    print(f"  Device  : {device}")
    print(f"  Output  : {output_path}")
    print()

    # Validate inputs
    for path, name in [(video_path, "Video"), (model_path, "Model"),
                        (holistic_path, "Holistic model")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    # Pipeline
    frames, landmarks, raw_landmarks, fps      = extract_landmarks(video_path, holistic_path)
    windows_masked, windows_full, window_starts = build_kinetic_windows(landmarks)

    if len(windows_masked) == 0:
        print("ERROR: Video too short — need at least 60 frames (2 seconds).")
        sys.exit(1)

    recon_windows                = run_inference(windows_masked, model_path, device)
    mse, rmse                    = compute_reconstruction_mse(windows_full, recon_windows)
    build_output_video(frames, landmarks, raw_landmarks, recon_windows, window_starts, fps, output_path)

    print(f"\nDone.")
    print(f"  MSE  : {mse:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  Video: {output_path}")


if __name__ == "__main__":
    main()
