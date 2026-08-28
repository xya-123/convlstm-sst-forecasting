import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from sst_forecasting.data import SSTScaler, build_datasets
from sst_forecasting.metrics import (
    MetricAccumulator,
    masked_change_anomaly_mse,
    masked_mse,
    masked_spatial_mean,
)
from sst_forecasting.models import (
    CNNForecaster,
    ConvLSTMForecaster,
    ResidualConvLSTMForecaster,
    build_model,
)
from prepare_data import collect_daily_files
from train import (
    forecast_loss_components,
    make_lr_scheduler,
    metric_improved,
    update_early_stopping_wait,
)


class DataTests(unittest.TestCase):
    def test_daily_files_are_sorted_by_full_date(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            names = [
                "2020/10/oisst.20200110.nc",
                "2020/2/oisst.20200109.nc",
                "2020/1/oisst.20200108.nc",
            ]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            ordered = collect_daily_files(root, allow_gaps=False)
            self.assertEqual(
                [day.isoformat() for day, _ in ordered],
                ["2020-01-08", "2020-01-09", "2020-01-10"],
            )

    def test_scaler_uses_training_values(self) -> None:
        train = np.asarray([[[1.0]], [[3.0]], [[np.nan]]], dtype=np.float32)
        scaler = SSTScaler("minmax").fit(train)
        self.assertEqual(scaler.offset, 1.0)
        self.assertEqual(scaler.scale, 2.0)
        transformed = scaler.transform(np.asarray([[[2.0]], [[np.nan]]], dtype=np.float32))
        self.assertAlmostEqual(float(transformed[0, 0, 0]), 0.5)
        self.assertTrue(np.isnan(transformed[1, 0, 0]))

    def test_window_shapes_and_target_splits(self) -> None:
        data = np.random.default_rng(3).normal(size=(20, 8, 8)).astype(np.float32)
        data[:, :2, :2] = np.nan
        datasets, _, split = build_datasets(
            data, seq_len=3, train_ratio=0.6, val_ratio=0.2, normalization="zscore"
        )
        x, y, mask, target_index = datasets["train"][0]
        self.assertEqual(tuple(x.shape), (3, 1, 8, 8))
        self.assertEqual(tuple(y.shape), (1, 8, 8))
        self.assertEqual(tuple(mask.shape), (1, 8, 8))
        self.assertEqual(int(target_index), 3)
        self.assertEqual(int(datasets["val"].target_indices[0]), split["train_end"])
        self.assertEqual(int(datasets["test"].target_indices[0]), split["val_end"])


class ModelTests(unittest.TestCase):
    def test_convlstm_output_shape(self) -> None:
        model = ConvLSTMForecaster(input_dim=1, hidden_dims=(4, 4), kernel_size=3)
        output = model(torch.randn(2, 3, 1, 8, 8))
        self.assertEqual(tuple(output.shape), (2, 1, 8, 8))

    def test_cnn_output_shape(self) -> None:
        model = CNNForecaster(seq_len=3, input_dim=1, hidden_dim=4)
        output = model(torch.randn(2, 3, 1, 8, 8))
        self.assertEqual(tuple(output.shape), (2, 1, 8, 8))

    def test_regression_head_is_not_tanh_bounded(self) -> None:
        model = ConvLSTMForecaster(input_dim=1, hidden_dims=(2,), kernel_size=3)
        with torch.no_grad():
            model.readout.weight.zero_()
            model.readout.bias.fill_(20.0)
        output = model(torch.zeros(1, 2, 1, 4, 4))
        self.assertTrue(torch.allclose(output, torch.full_like(output, 20.0)))

    def test_residual_convlstm_starts_as_persistence(self) -> None:
        model = ResidualConvLSTMForecaster(input_dim=1, hidden_dims=(2,), kernel_size=3)
        sequence = torch.randn(2, 3, 1, 4, 4)
        output = model(sequence)
        self.assertTrue(torch.allclose(output, sequence[:, -1]))

    def test_residual_model_can_be_built_by_name(self) -> None:
        model = build_model(
            "residual-convlstm", input_dim=1, hidden_dims=(2,), kernel_size=3
        )
        self.assertIsInstance(model, ResidualConvLSTMForecaster)

    def test_small_random_readout_starts_backbone_gradient_immediately(self) -> None:
        torch.manual_seed(7)
        model = ResidualConvLSTMForecaster(
            input_dim=1,
            hidden_dims=(2,),
            kernel_size=3,
            readout_init_std=1e-3,
        )
        sequence = torch.randn(2, 3, 1, 4, 4)
        target = torch.randn(2, 1, 4, 4)
        loss = torch.nn.functional.mse_loss(model(sequence), target)
        loss.backward()
        backbone_gradient = model.cells[0].gates.weight.grad
        self.assertIsNotNone(backbone_gradient)
        self.assertGreater(float(backbone_gradient.abs().sum()), 0.0)
        self.assertTrue(torch.allclose(model.readout.bias, torch.zeros_like(model.readout.bias)))

    def test_residual_readout_init_is_stored_in_model_config(self) -> None:
        model = ResidualConvLSTMForecaster(readout_init_std=1e-3)
        self.assertEqual(model.model_config["readout_init_std"], 1e-3)


class MetricTests(unittest.TestCase):
    def test_masked_mse_ignores_land(self) -> None:
        prediction = torch.tensor([[[[2.0, 100.0]]]])
        target = torch.tensor([[[[1.0, 0.0]]]])
        mask = torch.tensor([[[[True, False]]]])
        self.assertAlmostEqual(float(masked_mse(prediction, target, mask)), 1.0)

    def test_masked_spatial_mean_is_computed_per_sample(self) -> None:
        values = torch.tensor([[[[1.0, 3.0]]], [[[10.0, 20.0]]]])
        mask = torch.tensor([[[[True, True]]], [[[True, False]]]])
        means = masked_spatial_mean(values, mask)
        self.assertEqual(tuple(means.shape), (2, 1, 1, 1))
        self.assertTrue(torch.allclose(means.flatten(), torch.tensor([2.0, 10.0])))

    def test_change_anomaly_loss_ignores_uniform_daily_correction(self) -> None:
        persistence = torch.zeros(1, 1, 1, 3)
        target = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
        prediction = target + 2.0
        mask = torch.ones_like(target, dtype=torch.bool)
        loss = masked_change_anomaly_mse(prediction, target, persistence, mask)
        self.assertAlmostEqual(float(loss), 0.0)
        self.assertAlmostEqual(float(masked_mse(prediction, target, mask)), 4.0)

    def test_change_anomaly_loss_detects_missing_local_pattern(self) -> None:
        persistence = torch.zeros(1, 1, 1, 3)
        target = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
        prediction = torch.zeros_like(target)
        mask = torch.ones_like(target, dtype=torch.bool)
        loss = masked_change_anomaly_mse(prediction, target, persistence, mask)
        self.assertAlmostEqual(float(loss), 2.0 / 3.0)

    def test_accumulator_compares_model_and_persistence_on_same_mask(self) -> None:
        prediction = torch.tensor([[[[2.0, 100.0]]]])
        target = torch.tensor([[[[1.0, 0.0]]]])
        persistence = torch.tensor([[[[3.0, -100.0]]]])
        mask = torch.tensor([[[[True, False]]]])
        accumulator = MetricAccumulator()
        accumulator.update(prediction, target, persistence, mask)
        result = accumulator.compute()
        self.assertAlmostEqual(result["rmse_celsius"], 1.0)
        self.assertAlmostEqual(result["bias_celsius"], 1.0)
        self.assertAlmostEqual(result["persistence_rmse_celsius"], 2.0)
        self.assertAlmostEqual(result["skill_vs_persistence"], 0.5)
        self.assertEqual(result["ocean_pixel_count"], 1)

    def test_daily_change_diagnostics_detect_scale_and_correlation(self) -> None:
        persistence = torch.zeros(1, 1, 1, 4)
        target = torch.tensor([[[[-2.0, -1.0, 1.0, 2.0]]]])
        prediction = target * 0.5
        mask = torch.ones_like(target, dtype=torch.bool)
        accumulator = MetricAccumulator()
        accumulator.update(prediction, target, persistence, mask)
        result = accumulator.compute()
        self.assertAlmostEqual(result["daily_change_correlation"], 1.0)
        self.assertAlmostEqual(result["daily_change_variability_ratio"], 0.5)
        self.assertAlmostEqual(result["observed_daily_change_mean_celsius"], 0.0)
        self.assertAlmostEqual(result["predicted_daily_change_mean_celsius"], 0.0)

    def test_daily_change_correlation_is_nan_for_constant_prediction(self) -> None:
        persistence = torch.zeros(1, 1, 1, 3)
        target = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
        prediction = torch.zeros_like(target)
        mask = torch.ones_like(target, dtype=torch.bool)
        accumulator = MetricAccumulator()
        accumulator.update(prediction, target, persistence, mask)
        result = accumulator.compute()
        self.assertTrue(math.isnan(result["daily_change_correlation"]))


class TrainingUtilityTests(unittest.TestCase):
    def test_metric_improvement_handles_modes_and_nan(self) -> None:
        self.assertTrue(metric_improved(0.9, 1.0, "min"))
        self.assertTrue(metric_improved(0.2, 0.1, "max"))
        self.assertFalse(metric_improved(float("nan"), 1.0, "min"))

    def test_min_epochs_delays_patience_count(self) -> None:
        wait = 0
        for epoch in range(1, 50):
            wait = update_early_stopping_wait(
                epoch, min_epochs=50, objective_improved=False, current_wait=wait
            )
        self.assertEqual(wait, 0)

        for epoch in range(50, 69):
            wait = update_early_stopping_wait(
                epoch, min_epochs=50, objective_improved=False, current_wait=wait
            )
        self.assertEqual(wait, 19)
        wait = update_early_stopping_wait(
            69, min_epochs=50, objective_improved=False, current_wait=wait
        )
        self.assertEqual(wait, 20)

    def test_objective_improvement_resets_early_stopping_wait(self) -> None:
        wait = update_early_stopping_wait(
            55, min_epochs=50, objective_improved=True, current_wait=7
        )
        self.assertEqual(wait, 0)

    def test_composite_forecast_loss_respects_anomaly_weight(self) -> None:
        persistence = torch.zeros(1, 1, 1, 3)
        target = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
        prediction = torch.zeros_like(target)
        mask = torch.ones_like(target, dtype=torch.bool)

        unweighted = forecast_loss_components(
            prediction, target, persistence, mask, change_anomaly_weight=0.0
        )
        weighted = forecast_loss_components(
            prediction, target, persistence, mask, change_anomaly_weight=1.0
        )

        self.assertAlmostEqual(
            float(unweighted["objective"]), float(unweighted["forecast_mse"])
        )
        self.assertAlmostEqual(
            float(weighted["objective"]),
            float(weighted["forecast_mse"] + weighted["change_anomaly_mse"]),
        )

    def test_plateau_scheduler_reduces_learning_rate(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=3e-4)
        args = SimpleNamespace(
            lr_scheduler="plateau",
            lr_factor=0.5,
            lr_patience=0,
            min_learning_rate=1e-5,
        )
        scheduler = make_lr_scheduler(optimizer, args)
        self.assertIsNotNone(scheduler)
        scheduler.step(1.0)
        scheduler.step(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.5e-4)


if __name__ == "__main__":
    unittest.main()
