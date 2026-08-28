from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from sst_forecasting.data import SSTScaler, SSTWindowDataset, load_sst_data
from sst_forecasting.metrics import MetricAccumulator
from sst_forecasting.models import build_model
from sst_forecasting.utils import ensure_directory, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an SST forecasting checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-examples", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def plot_examples(examples: list[dict[str, object]], path: Path) -> None:
    if not examples:
        return
    figure, axes = plt.subplots(len(examples), 4, figsize=(15, 3.6 * len(examples)), squeeze=False)
    for row, example in enumerate(examples):
        mask = np.asarray(example["mask"], dtype=bool)
        previous = np.where(mask, np.asarray(example["previous"]), np.nan)
        target = np.where(mask, np.asarray(example["target"]), np.nan)
        prediction = np.where(mask, np.asarray(example["prediction"]), np.nan)
        error = np.where(mask, np.abs(prediction - target), np.nan)

        temperature_values = np.concatenate(
            [previous[np.isfinite(previous)], target[np.isfinite(target)], prediction[np.isfinite(prediction)]]
        )
        vmin, vmax = np.percentile(temperature_values, [1, 99])
        panels = (
            (previous, "Previous day", "RdYlBu_r", vmin, vmax),
            (target, "Ground truth", "RdYlBu_r", vmin, vmax),
            (prediction, "Model prediction", "RdYlBu_r", vmin, vmax),
            (
                error,
                "Absolute error",
                "magma",
                0.0,
                max(float(np.nanpercentile(error, 99)), 1e-6),
            ),
        )
        for column, (image, title, cmap, lower, upper) in enumerate(panels):
            axis = axes[row, column]
            artist = axis.imshow(image, origin="lower", cmap=cmap, vmin=lower, vmax=upper)
            axis.set_title(f"{title}\n{example['date']}")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04, label="°C")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    data_path = args.data or Path(checkpoint["data_path"])
    start_date = args.start_date if args.start_date is not None else checkpoint.get("start_date")
    loaded = load_sst_data(data_path, start_date)
    split = checkpoint["split"]
    if int(split["n_days"]) != len(loaded.sst):
        raise ValueError(
            f"Checkpoint expects {split['n_days']} days, but {data_path} contains {len(loaded.sst)}."
        )

    sst_scaler = SSTScaler.from_state_dict(checkpoint["scaler"])
    scaled_sst = sst_scaler.transform(loaded.sst)
    test_dataset = SSTWindowDataset(
        scaled_sst,
        seq_len=int(split["seq_len"]),
        target_start=int(split["val_end"]),
        target_end=int(split["n_days"]),
    )
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(checkpoint["model_name"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    metrics = MetricAccumulator()
    examples: list[dict[str, object]] = []
    for x, y, ocean_mask, target_indices in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ocean_mask = ocean_mask.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(x)

        prediction_real = sst_scaler.inverse_tensor(prediction.float())
        target_real = sst_scaler.inverse_tensor(y.float())
        persistence_real = sst_scaler.inverse_tensor(x[:, -1].float())
        metrics.update(prediction_real, target_real, persistence_real, ocean_mask)

        remaining = max(args.save_examples - len(examples), 0)
        for index in range(min(remaining, x.shape[0])):
            target_index = int(target_indices[index].item())
            examples.append(
                {
                    "date": str(loaded.dates[target_index]),
                    "previous": persistence_real[index, 0].cpu().numpy(),
                    "target": target_real[index, 0].cpu().numpy(),
                    "prediction": prediction_real[index, 0].cpu().numpy(),
                    "mask": ocean_mask[index, 0].cpu().numpy(),
                }
            )

    result = metrics.compute()
    result.update(
        {
            "checkpoint": str(args.checkpoint),
            "data": str(data_path),
            "model": checkpoint["model_name"],
            "normalization": checkpoint["normalization"],
            "loss_mask": checkpoint["loss_mask"],
            "best_epoch": int(checkpoint["best_epoch"]),
        }
    )

    output_dir = ensure_directory(args.output_dir or args.checkpoint.parent / "evaluation")
    write_json(output_dir / "metrics.json", result)
    if examples:
        plot_examples(examples, output_dir / "examples.png")

    print(f"Model RMSE:       {result['rmse_celsius']:.4f} degC")
    print(f"Model MAE:        {result['mae_celsius']:.4f} degC")
    print(f"Persistence RMSE: {result['persistence_rmse_celsius']:.4f} degC")
    print(f"Skill:            {result['skill_vs_persistence']:.4f}")
    print(f"Saved evaluation: {output_dir}")


if __name__ == "__main__":
    main()
