import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SkeletonEncoder(nn.Module):
    """
    Dense Temporal Encoder for masked skeleton reconstruction.

    Shape contract (NO global pooling anywhere):
        Input  : (B, T, V, C)   e.g. (B, 60, 543, 3)
        Encoded: (B, T, hidden)  — temporal dim is FULLY PRESERVED
        Output : (B, T, V, C)   — pointwise reconstruction per frame

    Frame 5 of the latent is used ONLY to predict Frame 5 coordinates.
    The Reconstruction Head is applied identically to every time-step via
    the Linear layers, which act as a shared 1×1 convolution over time.
    """

    def __init__(self, num_nodes=543, in_channels=3, hidden_dim=512,
                 num_layers=4, nhead=8):
        super(SkeletonEncoder, self).__init__()
        self.num_nodes   = num_nodes
        self.in_channels = in_channels
        self.input_dim   = num_nodes * in_channels  # 1629

        # ── Spatial Projection ────────────────────────────────────────────────
        # Maps (B, T, 1629) → (B, T, hidden_dim).  NO pooling over T.
        self.spatial_proj = nn.Linear(self.input_dim, hidden_dim)

        # ── Positional Encoding ───────────────────────────────────────────────
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=60)

        # ── Dense Temporal Encoder (Transformer) ──────────────────────────────
        # batch_first=True → input/output shape: (B, T, hidden_dim).
        # dropout=0.5: heavy regularization forces the model to NOT rely on
        # any single attention head — it must distribute learning across all.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.5,             # Blindfold regularization
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False   # norm_first=True is incompatible with nested tensors
        )

        # ── Pointwise Reconstruction Head (Lean Bottleneck) ───────────────────
        # Bottleneck at 256 units prevents the model from hiding in a flat
        # local minimum via over-parameterization. It must compress the
        # hidden state down and reconstruct hand coords from scratch.
        BOTTLENECK = 256
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, BOTTLENECK),
            nn.GELU(),
            nn.Dropout(0.3),             # Additional regularization at decode
            nn.Linear(BOTTLENECK, BOTTLENECK),
            nn.GELU(),
            nn.Linear(BOTTLENECK, self.input_dim),
        )
        self._init_recon_head()

        # ── Orthogonal Initialization for Temporal Layers ─────────────────────
        # Orthogonal weights preserve signal norms over long sequences, keeping
        # motion gradients alive through all 60 frames.
        self._init_temporal_orthogonal()

    # ─────────────────────────────────────────────────────────────────────────
    def _init_recon_head(self):
        """Kaiming Normal for the coordinate regression head."""
        for layer in self.reconstruction_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def _init_temporal_orthogonal(self):
        """
        Orthogonal initialization for all Linear sub-layers inside the
        TransformerEncoder (attention projections + FFN).  Orthogonal
        matrices are norm-preserving: ||Wx|| = ||x||, so gradients neither
        explode nor vanish when back-propagating through 60 attention steps.
        """
        for module in self.temporal_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Also apply to spatial projection (feeds into temporal stack)
        nn.init.orthogonal_(self.spatial_proj.weight)
        nn.init.zeros_(self.spatial_proj.bias)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x):
        """
        x : (B, T, V, C)   e.g. (B, 60, 543, 3) — masked skeleton windows
        Returns reconstructed tensor of the same shape.
        """
        B, T, V, C = x.size()

        # (B, T, V, C) → (B, T, V*C)  — spatial flatten, T is untouched
        x_flat = x.view(B, T, V * C)

        # (B, T, V*C) → (B, T, hidden_dim)
        x = self.spatial_proj(x_flat)

        # Add per-frame positional signal
        x = self.pos_encoder(x)

        # Dense temporal attention: (B, T, hidden) → (B, T, hidden)
        # Shape is FULLY PRESERVED — no pooling, no squeeze over T
        x = self.temporal_encoder(x)

        # Pointwise decode: each frame's hidden state → that frame's coords
        # Linear operates on last dim only, so T is still preserved
        out = self.reconstruction_head(x)  # (B, T, V*C)

        # Reshape to original spatial structure
        return out.view(B, T, V, C)        # (B, T, V, C)


# ─── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = SkeletonEncoder()
    dummy  = torch.randn(2, 60, 543, 3)
    out    = model(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    assert out.shape == dummy.shape, "Shape mismatch!"
    print("Shape contract OK — temporal dim fully preserved.")
