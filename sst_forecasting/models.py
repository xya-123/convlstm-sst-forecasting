from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """One ConvLSTM recurrence step for 2-D feature maps."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial dimensions.")
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def initial_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (x.shape[0], self.hidden_dim, x.shape[-2], x.shape[-1])
        return x.new_zeros(shape), x.new_zeros(shape)


class ConvLSTMForecaster(nn.Module):
    """Stacked ConvLSTM encoder plus an unrestricted 1x1 regression head."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: Sequence[int] = (16, 16),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if not hidden_dims or any(value < 1 for value in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")

        cells: list[ConvLSTMCell] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            cells.append(ConvLSTMCell(current_dim, hidden_dim, kernel_size))
            current_dim = hidden_dim
        self.cells = nn.ModuleList(cells)
        self.readout = nn.Conv2d(hidden_dims[-1], 1, kernel_size=1)
        self.model_config = {
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "kernel_size": kernel_size,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], received {tuple(x.shape)}.")

        layer_input = x
        for cell in self.cells:
            h, c = cell.initial_state(layer_input[:, 0])
            outputs = []
            for time_index in range(layer_input.shape[1]):
                h, c = cell(layer_input[:, time_index], (h, c))
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)
        return self.readout(layer_input[:, -1])


class ResidualConvLSTMForecaster(ConvLSTMForecaster):
    """Predict the change from the latest observation instead of the full SST field.

    The returned tensor is still the next-day SST in normalized units.  Only the
    internal parameterization changes: the ConvLSTM readout represents a daily
    increment, which is added to the final input frame.  Zero-initializing the
    readout makes the initial network exactly equal to persistence forecasting.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: Sequence[int] = (16, 16),
        kernel_size: int = 3,
    ) -> None:
        if input_dim != 1:
            raise ValueError("ResidualConvLSTMForecaster currently requires input_dim=1.")
        super().__init__(input_dim, hidden_dims, kernel_size)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = super().forward(x)
        return x[:, -1] + correction


class CNNForecaster(nn.Module):
    """Baseline that treats the temporal dimension as input channels."""

    def __init__(self, seq_len: int = 10, input_dim: int = 1, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(seq_len * input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.model_config = {
            "seq_len": seq_len,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], received {tuple(x.shape)}.")
        batch, time, channels, height, width = x.shape
        return self.network(x.reshape(batch, time * channels, height, width))


def build_model(name: str, **kwargs: object) -> nn.Module:
    if name == "convlstm":
        return ConvLSTMForecaster(**kwargs)
    if name == "residual-convlstm":
        return ResidualConvLSTMForecaster(**kwargs)
    if name == "cnn":
        return CNNForecaster(**kwargs)
    raise ValueError("Unknown model. Choose 'convlstm', 'residual-convlstm', or 'cnn'.")
