import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

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
    batch_size      = 32
    learning_rate   = 1e-4
    epochs          = 30
    lambda_vel      = 0.5    # weight for velocity loss
    lambda_smooth   = 0.1    # weight for temporal smoothness loss
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Input channels: 9  (position=3 | velocity=3 | acceleration=3)")
    print(f"Loss weights   : recon=1.0 | vel={lambda_vel} | smooth={lambda_smooth}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset    = SkeletonDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SkeletonEncoder().to(device)   # defaults: in_channels=9, nhead=16
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print("Training from scratch — v5 Kinetic Engine (9-channel, 16-head).")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = GradScaler('cuda')

    # ── Tracking ──────────────────────────────────────────────────────────────
    log_total  = []
    log_recon  = []
    log_vel    = []
    log_smooth = []

    print(f"Starting v5 Kinetic Engine — {epochs} epochs...")
    print("─" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        run_total = run_recon = run_vel = run_smooth = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)
        for masked_inputs, targets in pbar:
            masked_inputs = masked_inputs.to(device)   # (B, 60, 543, 9)
            targets       = targets.to(device)         # (B, 60, 543, 9)

            optimizer.zero_grad()

            with autocast('cuda'):
                reconstructed = model(masked_inputs)   # (B, 60, 543, 9)

                # Hand landmark slice (indices 501:543)
                pred_hands   = reconstructed[:, :, 501:543, :]  # (B, 60, 42, 9)
                target_hands = targets[:, :, 501:543, :]

                # ── 1. Position Reconstruction Loss ───────────────────────────
                # Channels 0:3 = (x, y, z) — primary spatial objective
                recon_loss = criterion(pred_hands[..., 0:3], target_hands[..., 0:3])

                # ── 2. Velocity Loss ──────────────────────────────────────────
                # Channels 3:6 = smoothed velocity — speed and direction of hands
                vel_loss = criterion(pred_hands[..., 3:6], target_hands[..., 3:6])

                # ── 3. Temporal Smoothness Penalty ────────────────────────────
                # Penalize discontinuities in predicted position across frames.
                # Uses predicted position channels (0:3), not output vel channels,
                # to double-check motion coherence from the raw prediction.
                pred_pos    = pred_hands[..., 0:3]   # (B, 60, 42, 3)
                target_pos  = target_hands[..., 0:3]
                pred_delta  = pred_pos[:, 1:] - pred_pos[:, :-1]    # (B, 59, 42, 3)
                target_delta = target_pos[:, 1:] - target_pos[:, :-1]
                smooth_loss = criterion(pred_delta, target_delta)

                # ── Total Loss ────────────────────────────────────────────────
                loss = recon_loss + lambda_vel * vel_loss + lambda_smooth * smooth_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            run_total  += loss.item()
            run_recon  += recon_loss.item()
            run_vel    += vel_loss.item()
            run_smooth += smooth_loss.item()

            pbar.set_postfix({
                "tot":  f"{loss.item():.4f}",
                "pos":  f"{recon_loss.item():.4f}",
                "vel":  f"{vel_loss.item():.4f}",
                "smt":  f"{smooth_loss.item():.4f}",
            })

        n = len(dataloader)
        avg_total  = run_total  / n
        avg_recon  = run_recon  / n
        avg_vel    = run_vel    / n
        avg_smooth = run_smooth / n
        current_lr = scheduler.get_last_lr()[0]

        log_total.append(avg_total)
        log_recon.append(avg_recon)
        log_vel.append(avg_vel)
        log_smooth.append(avg_smooth)

        print(f"Epoch [{epoch:02d}/{epochs}] | "
              f"Total: {avg_total:.6f} | "
              f"Pos: {avg_recon:.6f} | "
              f"Vel: {avg_vel:.6f} | "
              f"Smt: {avg_smooth:.6f} | "
              f"LR: {current_lr:.2e}")

        scheduler.step()

        if epoch % 5 == 0:
            ckpt = os.path.join(models_dir, f"ssl_encoder_v5_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  Checkpoint saved: {ckpt}")

    # ── Final Save ─────────────────────────────────────────────────────────────
    final_path = os.path.join(models_dir, "ssl_encoder_v5_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining Complete. Final model: {final_path}")

    # ── Loss Curves ───────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("v5 Kinetic Engine — 30-Epoch Loss Curves", fontsize=13)
    ep = range(1, epochs + 1)

    ax1.plot(ep, log_total,  marker='o', label='Total',    color='royalblue')
    ax1.plot(ep, log_recon,  marker='s', label='Position', color='tomato')
    ax1.plot(ep, log_vel,    marker='^', label='Velocity', color='darkorange')
    ax1.plot(ep, log_smooth, marker='D', label='Smooth',   color='seagreen')
    ax1.set_title("Linear Scale"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.4)

    ax2.semilogy(ep, log_total,  marker='o', label='Total',    color='royalblue')
    ax2.semilogy(ep, log_recon,  marker='s', label='Position', color='tomato')
    ax2.semilogy(ep, log_vel,    marker='^', label='Velocity', color='darkorange')
    ax2.semilogy(ep, log_smooth, marker='D', label='Smooth',   color='seagreen')
    ax2.set_title("Log Scale"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss (log)")
    ax2.legend(); ax2.grid(True, alpha=0.4, which='both')

    plt.tight_layout()
    plot_path = os.path.join(base_dir, "loss_curve_v5.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Loss curve saved: {plot_path}")


if __name__ == "__main__":
    train()
