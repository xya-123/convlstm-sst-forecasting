import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from sst_forecasting.data import SSTScaler, build_datasets
from sst_forecasting.metrics import MetricAccumulator, masked_mse
from sst_forecasting.models import (
    CNNForecaster,
    ConvLSTMForecaster,
    ResidualConvLSTMForecaster,
    build_model,
)
from prepare_data import collect_daily_files


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


class MetricTests(unittest.TestCase):
    def test_masked_mse_ignores_land(self) -> None:
        prediction = torch.tensor([[[[2.0, 100.0]]]])
        target = torch.tensor([[[[1.0, 0.0]]]])
        mask = torch.tensor([[[[True, False]]]])
        self.assertAlmostEqual(float(masked_mse(prediction, target, mask)), 1.0)

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


if __name__ == "__main__":
    unittest.main()
