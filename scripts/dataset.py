import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d


# ─── Kinetic Feature Engineering ──────────────────────────────────────────────

def compute_kinetic_features(pos: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Build a 9-channel kinetic representation from raw 3D positions.

    Args:
        pos   : (T, V, 3) — raw landmark positions
        sigma : Gaussian smoothing sigma (applied to vel and acc to kill jitter)

    Returns:
        (T, V, 9) — [position(3) | velocity(3) | acceleration(3)]

    Channel layout:
        0:3  — raw position (x, y, z)
        3:6  — smoothed 1st-order velocity  (frame-to-frame delta)
        6:9  — smoothed 2nd-order acceleration (delta of velocity)
    """
    T, V, C = pos.shape

    # 1st-order: velocity — pad frame-0 with zeros so shape stays (T, V, C)
    vel_raw = np.zeros_like(pos)
    vel_raw[1:] = pos[1:] - pos[:-1]

    # 2nd-order: acceleration — pad frames 0,1 with zeros
    acc_raw = np.zeros_like(pos)
    acc_raw[2:] = vel_raw[2:] - vel_raw[1:-1]

    # Gaussian smooth along temporal axis (axis=0) to remove MediaPipe jitter
    vel = gaussian_filter1d(vel_raw, sigma=sigma, axis=0)
    acc = gaussian_filter1d(acc_raw, sigma=sigma, axis=0)

    # Concatenate on channel axis: (T, V, 9)
    return np.concatenate([pos, vel, acc], axis=-1)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SkeletonDataset(Dataset):
    def __init__(self, data_dir, window_size=60, sigma=1.5):
        self.data_dir    = data_dir
        self.window_size = window_size
        self.sigma       = sigma
        self.file_paths  = glob.glob(os.path.join(data_dir, "*.npy"))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load raw (60, 543, 3) position window
        pos = np.load(path)                    # (T, V, 3)

        # Build 9-channel kinetic feature tensor
        kinetic = compute_kinetic_features(pos, sigma=self.sigma)  # (T, V, 9)

        # Convert to torch
        kinetic = torch.tensor(kinetic, dtype=torch.float32)
        target  = kinetic.clone()

        # TOTAL BLINDFOLD: zero all 9 channels of hands (501:543) in every frame.
        # Position, velocity, and acceleration are all masked — the model gets
        # zero kinetic information about the hands and must infer from arms.
        masked = kinetic.clone()
        masked[:, 501:543, :] = 0.0   # (T, 42, 9) → all zeros

        return masked, target


# ─── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "Dataset", "training_chunks")

    if os.path.exists(data_dir):
        ds = SkeletonDataset(data_dir)
        if len(ds) > 0:
            masked, target = ds[0]
            print(f"Dataset Size     : {len(ds)}")
            print(f"Masked Shape     : {masked.shape}   (expected: 60, 543, 9)")
            print(f"Target Shape     : {target.shape}")
            # All 9 channels of hands must be zero in masked
            hand_sum = masked[:, 501:543, :].abs().sum().item()
            print(f"Hand energy in masked input: {hand_sum:.4f}  (must be 0.0)")
            # Velocity channels of body should be non-zero
            body_vel = masked[:, :501, 3:6].abs().sum().item()
            print(f"Body velocity energy  : {body_vel:.4f}  (must be > 0.0)")
