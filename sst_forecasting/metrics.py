from __future__ import annotations

from dataclasses import dataclass

import torch


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over selected pixels only."""

    mask = mask.to(dtype=prediction.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * mask).sum() / denominator


@dataclass
class MetricAccumulator:
    squared_error: float = 0.0
    absolute_error: float = 0.0
    persistence_squared_error: float = 0.0
    count: int = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        persistence: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        valid = mask.bool()
        error = prediction[valid] - target[valid]
        persistence_error = persistence[valid] - target[valid]
        self.squared_error += float((error**2).sum().item())
        self.absolute_error += float(error.abs().sum().item())
        self.persistence_squared_error += float((persistence_error**2).sum().item())
        self.count += int(valid.sum().item())

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("No valid ocean pixels were accumulated.")
        rmse = (self.squared_error / self.count) ** 0.5
        mae = self.absolute_error / self.count
        persistence_rmse = (self.persistence_squared_error / self.count) ** 0.5
        skill = 1.0 - rmse / persistence_rmse if persistence_rmse > 0 else float("nan")
        return {
            "rmse_celsius": rmse,
            "mae_celsius": mae,
            "persistence_rmse_celsius": persistence_rmse,
            "skill_vs_persistence": skill,
            "ocean_pixel_count": float(self.count),
        }
