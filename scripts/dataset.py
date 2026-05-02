import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, data_dir, window_size=60):
        self.data_dir    = data_dir
        self.window_size = window_size
        self.file_paths  = glob.glob(os.path.join(data_dir, "*.npy"))
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # Load the (60, 543, 3) window
        data = np.load(path)
        data = torch.tensor(data, dtype=torch.float32)

        # TOTAL BLINDFOLD: zero out hands (501:543) in ALL 60 frames.
        # The model must hallucinate hand positions purely from
        # Elbows/Shoulders (468:500), which remain fully visible.
        # Arms/Pose (0:500) and Face (0:468) are untouched.
        masked_data = data.clone()
        masked_data[:, 501:543, :] = 0.0  # 100% — no frame escapes

        return masked_data, data

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "Dataset", "training_chunks")

    if os.path.exists(data_dir):
        ds = SkeletonDataset(data_dir)
        if len(ds) > 0:
            masked, target = ds[0]
            print(f"Dataset Size  : {len(ds)}")
            print(f"Masked Shape  : {masked.shape}")
            print(f"Target Shape  : {target.shape}")
            # Verify: ALL frames should have zeroed hands
            zero_frames = (masked[:, 501:543, :].abs().sum(dim=(1, 2)) == 0).sum().item()
            print(f"Frames with hands zeroed: {zero_frames}/60  (must be 60)")
