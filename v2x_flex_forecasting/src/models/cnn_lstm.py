"""CNN-LSTM model for continuous-history EV flexibility forecasting."""
from __future__ import annotations

import torch
from torch import nn


class V2XCNNLSTM(nn.Module):
    """
    CNN-LSTM using the same seven-channel information set as the other models.

    Input:
        history      (B,L,3)
        history_time (B,L,4)

    Output:
        (B,H,3)
    """

    def __init__(
        self,
        n_features: int = 3,
        calendar_dim: int = 4,
        n_targets: int = 3,
        horizon: int = 96,
        conv_channels: int = 32,
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer so sequence "
                "length is preserved."
            )
        if lstm_layers < 1:
            raise ValueError("lstm_layers must be >= 1.")

        self.n_features = n_features
        self.calendar_dim = calendar_dim
        self.n_targets = n_targets
        self.horizon = horizon

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(
                n_features + calendar_dim,
                conv_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, horizon * n_targets),
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

        x = torch.cat([history, history_time], dim=-1)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        y = self.head(x)
        return y.view(-1, self.horizon, self.n_targets)


def build_cnn_lstm(**kwargs) -> V2XCNNLSTM:
    return V2XCNNLSTM(**kwargs)
