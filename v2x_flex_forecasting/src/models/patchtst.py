"""Compact PatchTST-style model for EV flexibility forecasting."""
from __future__ import annotations

import torch
from torch import nn


class V2XPatchTST(nn.Module):
    """
    Patch-based Transformer encoder for the fixed continuous-history input.

    This implementation patches the multivariate seven-channel sequence,
    projects each flattened patch to d_model, applies a Transformer encoder,
    mean-pools patch representations, and predicts the complete horizon.
    """

    def __init__(
        self,
        history_steps: int = 672,
        n_features: int = 3,
        calendar_dim: int = 4,
        n_targets: int = 3,
        horizon: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if patch_len > history_steps:
            raise ValueError("patch_len cannot exceed history_steps.")
        if patch_len < 1 or stride < 1:
            raise ValueError("patch_len and stride must be >= 1.")

        self.history_steps = history_steps
        self.n_features = n_features
        self.calendar_dim = calendar_dim
        self.n_targets = n_targets
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = 1 + (history_steps - patch_len) // stride

        input_dim = n_features + calendar_dim
        self.patch_projection = nn.Linear(
            patch_len * input_dim,
            d_model,
        )

        self.position = nn.Parameter(
            torch.zeros(1, self.n_patches, d_model)
        )
        nn.init.normal_(self.position, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(
            d_model,
            horizon * n_targets,
        )

    def forward(
        self,
        history: torch.Tensor,
        history_time: torch.Tensor,
    ) -> torch.Tensor:
        if history.ndim != 3 or history_time.ndim != 3:
            raise ValueError("history and history_time must both be rank-3.")
        if history.shape[:2] != history_time.shape[:2]:
            raise ValueError(
                "history and history_time must share batch/history dimensions."
            )
        if history.shape[1] != self.history_steps:
            raise ValueError(
                f"Expected history length {self.history_steps}; "
                f"got {history.shape[1]}."
            )

        x = torch.cat([history, history_time], dim=-1)

        patches = x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride,
        )
        # PyTorch: (B, n_patches, channels, patch_len)
        patches = patches.contiguous().reshape(
            x.shape[0],
            self.n_patches,
            -1,
        )

        z = self.patch_projection(patches)
        z = z + self.position
        z = self.encoder(z)
        z = self.norm(z)
        z = z.mean(dim=1)

        y = self.head(z)
        return y.view(-1, self.horizon, self.n_targets)


def build_patchtst(**kwargs) -> V2XPatchTST:
    return V2XPatchTST(**kwargs)
