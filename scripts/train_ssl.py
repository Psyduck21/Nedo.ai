import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Append the base path to import from scripts and models
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)

from scripts.dataset import SkeletonDataset
from models.skeleton_encoder import SkeletonEncoder

def train():
    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir   = os.path.join(base_dir, "Dataset", "training_chunks")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # ── Hyperparameters ────────────────────────────────────────────────────────
    batch_size     = 32
    learning_rate  = 1e-4
    epochs         = 20            # v4 Blindfold stress test
    lambda_smooth  = 0.1
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print("Masking: TOTAL BLINDFOLD (hands zeroed in 100% of frames)")
    print(f"Temporal Smoothness lambda: {lambda_smooth}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset    = SkeletonDataset(data_dir=data_dir)  # Total blindfold — no mask_ratio
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    # ── Model (fresh init — v4 Blindfold architecture) ────────────────────────
    model = SkeletonEncoder().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Training from scratch — Blindfold + Dropout 0.5 + Lean Bottleneck (256).")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                   weight_decay=1e-4)

    # Cosine decay: LR tapers from 1e-4 → 1e-6 across all 20 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Mixed precision for RTX 2050
    scaler = GradScaler('cuda')

    epoch_losses       = []
    epoch_recon_losses = []
    epoch_smooth_losses = []

    print(f"Starting v4 Blindfold Stress Test — {epochs} epochs...")
    print("─" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        running_total  = 0.0
        running_recon  = 0.0
        running_smooth = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)
        for masked_inputs, targets in progress_bar:
            masked_inputs = masked_inputs.to(device)
            targets       = targets.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                reconstructed = model(masked_inputs)

                # ── Masked-Only Reconstruction Loss ───────────────────────────
                # MSE only on the 42 hand landmarks (501:543).
                # Pointwise: loss is computed frame-by-frame automatically
                # because MSELoss averages over all dims including T.
                recon_loss = criterion(
                    reconstructed[:, :, 501:543],   # (B, T, 42, 3)
                    targets[:, :, 501:543]
                )

                # ── Temporal Smoothness Penalty ───────────────────────────────
                # Penalizes the model if predicted hand positions jump
                # discontinuously between consecutive frames.
                # We match the VELOCITY (delta between frames) of the
                # prediction to the ground truth velocity.
                # pred_vel  shape: (B, T-1, 42, 3)
                pred_vel   = reconstructed[:, 1:, 501:543] - reconstructed[:, :-1, 501:543]
                target_vel = targets[:, 1:, 501:543]       - targets[:, :-1, 501:543]
                smooth_loss = criterion(pred_vel, target_vel)

                # ── Total Loss ────────────────────────────────────────────────
                loss = recon_loss + lambda_smooth * smooth_loss

            # Backward + grad clip (prevents re-explosion like Epoch 24 v1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_total  += loss.item()
            running_recon  += recon_loss.item()
            running_smooth += smooth_loss.item()
            progress_bar.set_postfix({
                "total": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "smth":  f"{smooth_loss.item():.4f}",
            })

        avg_total  = running_total  / len(dataloader)
        avg_recon  = running_recon  / len(dataloader)
        avg_smooth = running_smooth / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]

        epoch_losses.append(avg_total)
        epoch_recon_losses.append(avg_recon)
        epoch_smooth_losses.append(avg_smooth)

        print(f"Epoch [{epoch:02d}/{epochs}] | "
              f"Total: {avg_total:.6f} | "
              f"Recon: {avg_recon:.6f} | "
              f"Smooth: {avg_smooth:.6f} | "
              f"LR: {current_lr:.2e}")

        scheduler.step()

        # Save every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(models_dir, f"ssl_encoder_v4_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # ── Final Save ─────────────────────────────────────────────────────────────
    final_path = os.path.join(models_dir, "ssl_encoder_v4_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining Complete. Final model: {final_path}")

    # ── Loss Curve ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("v4 Blindfold Stress Test — 20-Epoch Loss Curves", fontsize=13)

    ep_range = range(1, epochs + 1)

    ax1.plot(ep_range, epoch_losses,       marker='o', label='Total Loss',   color='royalblue')
    ax1.plot(ep_range, epoch_recon_losses, marker='s', label='Recon Loss',   color='tomato')
    ax1.plot(ep_range, epoch_smooth_losses,marker='^', label='Smooth Loss',  color='seagreen')
    ax1.set_title("All Loss Components")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.semilogy(ep_range, epoch_losses,        marker='o', label='Total Loss',   color='royalblue')
    ax2.semilogy(ep_range, epoch_recon_losses,  marker='s', label='Recon Loss',   color='tomato')
    ax2.semilogy(ep_range, epoch_smooth_losses, marker='^', label='Smooth Loss',  color='seagreen')
    ax2.set_title("Log Scale (gradient collapse visible here)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss (log)")
    ax2.legend()
    ax2.grid(True, alpha=0.4, which='both')

    plot_path = os.path.join(base_dir, "loss_curve_v4.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"Loss curve saved: {plot_path}")


if __name__ == "__main__":
    train()
