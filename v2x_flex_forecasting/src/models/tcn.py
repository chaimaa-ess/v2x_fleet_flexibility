"""TCN model for continuous-history EV flexibility forecasting."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class Chomp1d(nn.Module):
    """Remove right padding so a Conv1d is strictly causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class ResidualCausalBlock(nn.Module):
    """One-convolution residual causal block used in the selection TCN."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.chomp = Chomp1d(padding)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.conv(x)
        z = self.chomp(z)
        z = self.norm(z)
        z = self.activation(z)
        z = self.dropout(z)
        return self.activation(z + residual)


class V2XTCN(nn.Module):
    """
    Continuous-history TCN.

    The model receives flexibility history and aligned historical calendar
    covariates separately, concatenates them to seven channels, and produces
    the complete H x n_targets forecast from the final causal state.
    """

    def __init__(
        self,
        n_features: int = 3,
        calendar_dim: int = 4,
        n_targets: int = 3,
        horizon: int = 96,
        channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: Tuple[int, ...] = (
            1, 2, 4, 8, 16, 32, 64, 128, 256
        ),
        head_hidden: int = 128,
    ) -> None:
        super().__init__()

        if channels < 1:
            raise ValueError("channels must be >= 1.")
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2.")

        self.n_features = n_features
        self.calendar_dim = calendar_dim
        self.n_targets = n_targets
        self.horizon = horizon

        self.input_projection = nn.Conv1d(
            n_features + calendar_dim,
            channels,
            kernel_size=1,
        )

        self.blocks = nn.Sequential(
            *[
                ResidualCausalBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, horizon * n_targets),
        )

    def forward(
        self,
        history: torch.Tensor,
        history_time: torch.Tensor,
    ) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError(
                f"history must be (B,L,F); got {tuple(history.shape)}."
            )
        if history_time.ndim != 3:
            raise ValueError(
                "history_time must be (B,L,C); "
                f"got {tuple(history_time.shape)}."
            )
        if history.shape[:2] != history_time.shape[:2]:
            raise ValueError(
                "history and history_time must share batch/history dimensions."
            )
        if history.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} history features; "
                f"got {history.shape[-1]}."
            )
        if history_time.shape[-1] != self.calendar_dim:
            raise ValueError(
                f"Expected {self.calendar_dim} calendar features; "
                f"got {history_time.shape[-1]}."
            )

        x = torch.cat([history, history_time], dim=-1).transpose(1, 2)
        x = self.input_projection(x)
        x = self.blocks(x)
        x = x[:, :, -1]
        y = self.head(x)
        return y.view(-1, self.horizon, self.n_targets)


def build_tcn(**kwargs) -> V2XTCN:
    return V2XTCN(**kwargs)
