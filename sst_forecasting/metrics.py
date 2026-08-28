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
    signed_error: float = 0.0
    persistence_squared_error: float = 0.0
    persistence_absolute_error: float = 0.0
    observed_change_sum: float = 0.0
    predicted_change_sum: float = 0.0
    observed_change_squared_sum: float = 0.0
    predicted_change_squared_sum: float = 0.0
    change_cross_product_sum: float = 0.0
    count: int = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        persistence: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        valid = mask.bool()
        prediction_valid = prediction[valid].double()
        target_valid = target[valid].double()
        persistence_valid = persistence[valid].double()
        error = prediction_valid - target_valid
        persistence_error = persistence_valid - target_valid
        predicted_change = prediction_valid - persistence_valid
        observed_change = target_valid - persistence_valid
        self.squared_error += float((error**2).sum().item())
        self.absolute_error += float(error.abs().sum().item())
        self.signed_error += float(error.sum().item())
        self.persistence_squared_error += float((persistence_error**2).sum().item())
        self.persistence_absolute_error += float(persistence_error.abs().sum().item())
        self.observed_change_sum += float(observed_change.sum().item())
        self.predicted_change_sum += float(predicted_change.sum().item())
        self.observed_change_squared_sum += float((observed_change**2).sum().item())
        self.predicted_change_squared_sum += float((predicted_change**2).sum().item())
        self.change_cross_product_sum += float(
            (observed_change * predicted_change).sum().item()
        )
        self.count += int(valid.sum().item())

    def compute(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("No valid ocean pixels were accumulated.")
        rmse = (self.squared_error / self.count) ** 0.5
        mae = self.absolute_error / self.count
        bias = self.signed_error / self.count
        persistence_rmse = (self.persistence_squared_error / self.count) ** 0.5
        persistence_mae = self.persistence_absolute_error / self.count
        skill = 1.0 - rmse / persistence_rmse if persistence_rmse > 0 else float("nan")

        observed_change_mean = self.observed_change_sum / self.count
        predicted_change_mean = self.predicted_change_sum / self.count
        observed_change_variance = max(
            self.observed_change_squared_sum / self.count - observed_change_mean**2,
            0.0,
        )
        predicted_change_variance = max(
            self.predicted_change_squared_sum / self.count - predicted_change_mean**2,
            0.0,
        )
        observed_change_std = observed_change_variance**0.5
        predicted_change_std = predicted_change_variance**0.5
        covariance = (
            self.change_cross_product_sum / self.count
            - observed_change_mean * predicted_change_mean
        )
        correlation_denominator = observed_change_std * predicted_change_std
        change_correlation = float("nan")
        if correlation_denominator > 1e-12:
            change_correlation = max(
                min(covariance / correlation_denominator, 1.0),
                -1.0,
            )
        variability_ratio = (
            predicted_change_std / observed_change_std
            if observed_change_std > 1e-12
            else float("nan")
        )
        return {
            "rmse_celsius": rmse,
            "mae_celsius": mae,
            "bias_celsius": bias,
            "persistence_rmse_celsius": persistence_rmse,
            "persistence_mae_celsius": persistence_mae,
            "skill_vs_persistence": skill,
            "rmse_improvement_percent": 100.0 * skill,
            "observed_daily_change_mean_celsius": observed_change_mean,
            "predicted_daily_change_mean_celsius": predicted_change_mean,
            "observed_daily_change_std_celsius": observed_change_std,
            "predicted_daily_change_std_celsius": predicted_change_std,
            "daily_change_variability_ratio": variability_ratio,
            "daily_change_correlation": change_correlation,
            "daily_change_rmse_celsius": rmse,
            "ocean_pixel_count": self.count,
        }
