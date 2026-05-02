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
    v5 Kinetic Encoder — Dense Temporal Transformer with 9-channel input.

    Input channels (per landmark per frame):
        0:3  — position  (x, y, z)
        3:6  — velocity  (Gaussian-smoothed frame deltas)
        6:9  — acceleration (Gaussian-smoothed 2nd-order deltas)

    Shape contract (NO global pooling anywhere):
        Input  : (B, T, V, 9)   e.g. (B, 60, 543, 9)
        Encoded: (B, T, hidden)  — temporal dim is FULLY PRESERVED
        Output : (B, T, V, 9)   — pointwise prediction per frame

    The Reconstruction Head acts as a shared 1×1 conv over T:
        frame t hidden state → all 9 channels for frame t.
    """

    def __init__(self, num_nodes=543, in_channels=9, hidden_dim=512,
                 num_layers=4, nhead=16):
        super(SkeletonEncoder, self).__init__()
        assert hidden_dim % nhead == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})"

        self.num_nodes   = num_nodes
        self.in_channels = in_channels
        self.input_dim   = num_nodes * in_channels  # 543 * 9 = 4887

        # ── Spatial Projection ────────────────────────────────────────────────
        # Maps (B, T, 4887) → (B, T, hidden_dim).  NO pooling over T.
        self.spatial_proj = nn.Linear(self.input_dim, hidden_dim)

        # ── Positional Encoding ───────────────────────────────────────────────
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=60)

        # ── Dense Temporal Encoder (Transformer, 16 heads) ────────────────────
        # 16 heads × 32 head_dim = 512. Extra heads distribute attention across
        # position, velocity, and acceleration sub-spaces simultaneously.
        # dropout=0.3: lighter than v4 (0.5) — we have richer input features now
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.3,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False
        )

        # ── Pointwise Reconstruction Head (Lean Bottleneck) ───────────────────
        # Predicts all 9 output channels (pos + vel + acc) from hidden state.
        # 256-unit bottleneck forces compression; can't memorize, must generalize.
        BOTTLENECK = 256
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, BOTTLENECK),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(BOTTLENECK, BOTTLENECK),
            nn.GELU(),
            nn.Linear(BOTTLENECK, self.input_dim),
        )

        # ── Initialization ────────────────────────────────────────────────────
        self._init_recon_head()
        self._init_temporal_orthogonal()

    def _init_recon_head(self):
        """Kaiming Normal for the coordinate regression head."""
        for layer in self.reconstruction_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def _init_temporal_orthogonal(self):
        """
        Orthogonal init for all Linear layers in the temporal encoder.
        Norm-preserving: ‖Wx‖ = ‖x‖ — keeps kinetic signals alive over 60 steps.
        """
        for module in self.temporal_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.spatial_proj.weight)
        nn.init.zeros_(self.spatial_proj.bias)

    def forward(self, x):
        """
        x : (B, T, V, 9)  — masked kinetic tensor (hands zeroed)
        Returns: (B, T, V, 9) — reconstructed kinetic tensor
        """
        B, T, V, C = x.size()

        # Flatten spatial: (B, T, V*9)
        x_flat = x.view(B, T, V * C)

        # Project to hidden: (B, T, hidden_dim)
        x = self.spatial_proj(x_flat)

        # Temporal position signal
        x = self.pos_encoder(x)

        # Dense attention — shape preserved: (B, T, hidden_dim)
        x = self.temporal_encoder(x)

        # Pointwise decode: hidden[t] → all 9 channels for frame t
        out = self.reconstruction_head(x)  # (B, T, V*9)

        return out.view(B, T, V, C)         # (B, T, V, 9)


# ─── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = SkeletonEncoder()
    dummy  = torch.randn(2, 60, 543, 9)
    out    = model(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    assert out.shape == dummy.shape, "Shape mismatch!"
    print("Shape contract OK — (B, 60, 543, 9) preserved end-to-end.")
