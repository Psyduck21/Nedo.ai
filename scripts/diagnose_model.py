import torch
import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)
from models.skeleton_encoder import SkeletonEncoder

SHORTCUT_LEAK_THRESHOLD = 0.1  # Std must exceed this for the shortcut to be declared dead

def diagnose():
    print("=" * 60)
    print("   ML AUDIT: SHORTCUT LEAK VERIFICATION (v2 Model)")
    print("=" * 60)

    model = SkeletonEncoder()
    model_path = os.path.join(base_dir, "models", "ssl_encoder_v2_final.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        print(f"✅ Loaded: {model_path}")
    else:
        print(f"❌ Checkpoint not found: {model_path}")
        print("   Run train_ssl.py first to generate the v2 model.")
        return

    model.eval()

    # Dummy input — hands zeroed out, arms/pose visible (matching training condition)
    dummy_input = torch.randn(4, 60, 543, 3)   # batch=4 for stable Std estimate
    dummy_input[:, :, 501:543, :] = 0.0         # Mask hands (501–542)
    # Arms/Pose (468–500) are intentionally left intact

    print(f"\n{'─' * 60}")
    print("  LAYER-BY-LAYER DATA-FLOW ANALYSIS")
    print(f"{'─' * 60}")

    with torch.no_grad():
        B, T, V, C = dummy_input.size()
        x_flat = dummy_input.view(B, T, V * C)

        # Stage 1: Spatial projection
        feat_spatial = model.spatial_proj(x_flat)
        print(f"  Stage 1 | Spatial Proj    → Mean: {feat_spatial.mean().item():+.6f}, Std: {feat_spatial.std().item():.6f}")

        # Stage 2: Positional encoding
        feat_pos = model.pos_encoder(feat_spatial)
        print(f"  Stage 2 | Pos Encoding    → Mean: {feat_pos.mean().item():+.6f}, Std: {feat_pos.std().item():.6f}")

        # Stage 3: Temporal transformer
        feat_temporal = model.temporal_encoder(feat_pos)
        print(f"  Stage 3 | Temporal Enc    → Mean: {feat_temporal.mean().item():+.6f}, Std: {feat_temporal.std().item():.6f}")

        # Stage 4: Reconstruction head (the critical measurement)
        out_recon = model.reconstruction_head(feat_temporal)
        recon_std  = out_recon.std().item()
        recon_mean = out_recon.mean().item()
        print(f"  Stage 4 | Recon Head      → Mean: {recon_mean:+.6f}, Std: {recon_std:.6f}  ◀ KEY METRIC")

        # Ratio: how much of the signal comes from the recon head vs raw input
        input_std = x_flat.std().item()
        ratio = recon_std / input_std
        print(f"\n  Input Std               : {input_std:.6f}")
        print(f"  Recon Head Std          : {recon_std:.6f}")
        print(f"  Recon/Input Std Ratio   : {ratio:.4f}  (baseline 1.0 = full expression)")

    print(f"\n{'─' * 60}")
    print("  VERDICT")
    print(f"{'─' * 60}")
    print(f"  Previous (v1, shortcut)  Recon Head Std : ~0.007")
    print(f"  Current  (v2, no shortcut) Recon Head Std : {recon_std:.6f}")
    print(f"  Threshold for 'Shortcut Dead'            : > {SHORTCUT_LEAK_THRESHOLD}")

    if recon_std > SHORTCUT_LEAK_THRESHOLD:
        print(f"\n  ✅ PASS — SHORTCUT LEAK IS DEAD.")
        print(f"  The Reconstruction Head Std ({recon_std:.4f}) exceeds the threshold ({SHORTCUT_LEAK_THRESHOLD}).")
        print(f"  The model is generating coordinates from scratch. Hallucination is active.")
    else:
        print(f"\n  ❌ FAIL — SHORTCUT LEAK MAY PERSIST.")
        print(f"  Recon Head Std ({recon_std:.4f}) is still below threshold ({SHORTCUT_LEAK_THRESHOLD}).")
        print(f"  Check skeleton_encoder.py for residual skip connections.")

    print("=" * 60)

if __name__ == "__main__":
    diagnose()
