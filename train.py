from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from sst_forecasting.data import SSTScaler, build_datasets, load_sst_data
from sst_forecasting.metrics import (
    MetricAccumulator,
    masked_change_anomaly_mse,
    masked_mse,
)
from sst_forecasting.models import build_model
from sst_forecasting.utils import ensure_directory, resolve_device, set_reproducible_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SST forecasting model.")
    parser.add_argument("--data", type=Path, default=Path("data.npy"))
    parser.add_argument("--start-date", default=None, help="Required for dated legacy .npy files.")
    parser.add_argument(
        "--model",
        choices=("convlstm", "residual-convlstm", "cnn"),
        default="convlstm",
    )
    parser.add_argument("--normalization", choices=("none", "minmax", "zscore"), default="minmax")
    parser.add_argument("--loss-mask", choices=("ocean", "all"), default="ocean")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[16, 16])
    parser.add_argument("--cnn-hidden-dim", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument(
        "--residual-readout-init-std",
        type=float,
        default=0.0,
        help=(
            "Residual-head weight std. Zero exactly starts as persistence; a tiny "
            "positive value lets gradients reach ConvLSTM on the first batch."
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help=(
            "Minimum completed epochs before early-stopping patience starts. "
            "Zero preserves the original behavior."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "plateau"),
        default="none",
        help="Reduce the learning rate when validation loss stops improving.",
    )
    parser.add_argument("--lr-patience", type=int, default=6)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--change-anomaly-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for local daily-change anomaly MSE. Zero reproduces the original "
            "forecast-MSE objective; 1.0 gives equal weight to the local-pattern term."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def make_model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    if args.model == "convlstm":
        kwargs: dict[str, Any] = {
            "input_dim": 1,
            "hidden_dims": args.hidden_dims,
            "kernel_size": args.kernel_size,
        }
    elif args.model == "residual-convlstm":
        kwargs = {
            "input_dim": 1,
            "hidden_dims": args.hidden_dims,
            "kernel_size": args.kernel_size,
            "readout_init_std": args.residual_readout_init_std,
        }
    else:
        kwargs = {
            "seq_len": args.seq_len,
            "input_dim": 1,
            "hidden_dim": args.cnn_hidden_dim,
        }
    return build_model(args.model, **kwargs), kwargs


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if args.lr_scheduler == "none":
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=1e-10,
        threshold_mode="abs",
        min_lr=args.min_learning_rate,
    )


def selected_mask(ocean_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "ocean":
        return ocean_mask
    return torch.ones_like(ocean_mask, dtype=torch.bool)


def forecast_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    persistence: torch.Tensor,
    mask: torch.Tensor,
    change_anomaly_weight: float,
) -> dict[str, torch.Tensor]:
    forecast_mse = masked_mse(prediction, target, mask)
    change_anomaly_mse = masked_change_anomaly_mse(
        prediction, target, persistence, mask
    )
    objective = forecast_mse + change_anomaly_weight * change_anomaly_mse
    return {
        "objective": objective,
        "forecast_mse": forecast_mse,
        "change_anomaly_mse": change_anomaly_mse,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    amp_scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    loss_mask_mode: str,
    change_anomaly_weight: float,
) -> dict[str, float]:
    model.train()
    accumulated = {
        "objective": 0.0,
        "forecast_mse": 0.0,
        "change_anomaly_mse": 0.0,
    }
    selected_count = 0

    for x, y, ocean_mask, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ocean_mask = ocean_mask.to(device, non_blocking=True)
        mask = selected_mask(ocean_mask, loss_mask_mode)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(x)
            losses = forecast_loss_components(
                prediction,
                y,
                x[:, -1],
                mask,
                change_anomaly_weight,
            )
        amp_scaler.scale(losses["objective"]).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()

        count = int(mask.sum().item())
        for name, loss in losses.items():
            accumulated[name] += float(loss.item()) * count
        selected_count += count

    return {
        name: value / max(selected_count, 1) for name, value in accumulated.items()
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    sst_scaler: SSTScaler,
    device: torch.device,
    use_amp: bool,
    loss_mask_mode: str,
    change_anomaly_weight: float,
) -> tuple[dict[str, float], dict[str, float | int]]:
    model.eval()
    accumulated = {
        "objective": 0.0,
        "forecast_mse": 0.0,
        "change_anomaly_mse": 0.0,
    }
    selected_count = 0
    metrics = MetricAccumulator()

    for x, y, ocean_mask, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ocean_mask = ocean_mask.to(device, non_blocking=True)
        mask = selected_mask(ocean_mask, loss_mask_mode)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(x)
            losses = forecast_loss_components(
                prediction,
                y,
                x[:, -1],
                mask,
                change_anomaly_weight,
            )

        count = int(mask.sum().item())
        for name, loss in losses.items():
            accumulated[name] += float(loss.item()) * count
        selected_count += count

        prediction_real = sst_scaler.inverse_tensor(prediction.float())
        target_real = sst_scaler.inverse_tensor(y.float())
        persistence_real = sst_scaler.inverse_tensor(x[:, -1].float())
        metrics.update(prediction_real, target_real, persistence_real, ocean_mask)

    averaged = {
        name: value / max(selected_count, 1) for name, value in accumulated.items()
    }
    return averaged, metrics.compute()


def create_loaders(
    datasets: dict[str, torch.utils.data.Dataset],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, DataLoader]:
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def save_history(path: Path, history: list[dict[str, float | int]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def metric_improved(value: float, best: float, mode: str) -> bool:
    """Return whether a finite validation value improves the tracked best."""

    if not math.isfinite(value):
        return False
    if mode == "min":
        return value < best - 1e-10
    if mode == "max":
        return value > best + 1e-10
    raise ValueError(f"Unknown metric comparison mode: {mode}")


def update_early_stopping_wait(
    epoch: int,
    min_epochs: int,
    objective_improved: bool,
    current_wait: int,
) -> int:
    """Count patience only after the minimum training period has been reached."""

    if objective_improved or epoch < min_epochs:
        return 0
    return current_wait + 1


def make_checkpoint(
    model: torch.nn.Module,
    args: argparse.Namespace,
    model_config: dict[str, Any],
    sst_scaler: SSTScaler,
    split: dict[str, int | float],
    epoch: int,
    criterion: str,
    val_losses: dict[str, float],
    val_metrics: dict[str, float | int],
) -> dict[str, Any]:
    """Build a self-describing evaluation checkpoint for one validation criterion."""

    return {
        "model_state": model.state_dict(),
        "model_name": args.model,
        "model_config": model_config,
        "scaler": sst_scaler.state_dict(),
        "split": split,
        "normalization": args.normalization,
        "loss_mask": args.loss_mask,
        "change_anomaly_weight": args.change_anomaly_weight,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "epochs_requested": args.epochs,
        "data_path": str(args.data),
        "start_date": args.start_date,
        "seed": args.seed,
        "checkpoint_criterion": criterion,
        "checkpoint_epoch": epoch,
        # Kept for compatibility with existing evaluation code and checkpoints.
        "best_epoch": epoch,
        "best_val_loss": val_losses["objective"],
        "best_val_forecast_mse": val_losses["forecast_mse"],
        "best_val_change_anomaly_mse": val_losses["change_anomaly_mse"],
        "best_val_metrics": val_metrics,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("batch-size, epochs, and patience must be positive.")
    if args.min_epochs < 0 or args.min_epochs > args.epochs:
        raise ValueError("min-epochs must be between 0 and epochs.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay cannot be negative.")
    if args.change_anomaly_weight < 0:
        raise ValueError("change-anomaly-weight cannot be negative.")
    if args.residual_readout_init_std < 0:
        raise ValueError("residual-readout-init-std cannot be negative.")
    if args.model != "residual-convlstm" and args.residual_readout_init_std != 0:
        raise ValueError(
            "residual-readout-init-std is only valid for --model residual-convlstm."
        )
    if args.lr_scheduler == "plateau":
        if args.min_learning_rate < 0 or args.min_learning_rate > args.learning_rate:
            raise ValueError(
                "min-learning-rate must be non-negative and cannot exceed learning-rate."
            )
        if args.lr_patience < 0 or not 0.0 < args.lr_factor < 1.0:
            raise ValueError(
                "lr-patience must be non-negative and lr-factor must be between 0 and 1."
            )

    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and not use_amp:
        print("AMP requested but CUDA is unavailable; continuing in float32.")

    loaded = load_sst_data(args.data, args.start_date)
    datasets, sst_scaler, split = build_datasets(
        loaded.sst,
        seq_len=args.seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        normalization=args.normalization,
    )
    loaders = create_loaders(datasets, args, device)
    model, model_config = make_model(args)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    lr_scheduler = make_lr_scheduler(optimizer, args)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_name = args.run_name or (
        f"{args.model}-{args.normalization}-t{args.seq_len}-seed{args.seed}"
    )
    run_dir = ensure_directory(args.output_root / run_name)
    config = vars(args).copy()
    config.update(
        {
            "data": str(args.data),
            "output_root": str(args.output_root),
            "run_dir": str(run_dir),
            "device_resolved": str(device),
            "amp_enabled": use_amp,
            "model_config": model_config,
            "split": split,
            "scaler": sst_scaler.state_dict(),
            "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
        }
    )
    write_json(run_dir / "config.json", config)

    print(f"Device: {device}; AMP: {use_amp}")
    print(f"Data shape: {loaded.sst.shape}; samples: {config['sample_counts']}")
    print(f"Scaler: {sst_scaler.state_dict()}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    history: list[dict[str, float | int]] = []
    criterion_modes = {
        "objective": "min",
        "rmse": "min",
        "anomaly": "min",
        "correlation": "max",
    }
    best_values = {
        "objective": math.inf,
        "rmse": math.inf,
        "anomaly": math.inf,
        "correlation": -math.inf,
    }
    best_epochs = {name: 0 for name in criterion_modes}
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_losses = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            amp_scaler,
            device,
            use_amp,
            args.loss_mask,
            args.change_anomaly_weight,
        )
        val_losses, val_metrics = validate(
            model,
            loaders["val"],
            sst_scaler,
            device,
            use_amp,
            args.loss_mask,
            args.change_anomaly_weight,
        )
        val_loss = val_losses["objective"]
        criterion_values = {
            "objective": val_loss,
            "rmse": float(val_metrics["rmse_celsius"]),
            "anomaly": val_losses["change_anomaly_mse"],
            "correlation": float(val_metrics["daily_change_correlation"]),
        }
        improvements = {
            name: metric_improved(value, best_values[name], criterion_modes[name])
            for name, value in criterion_values.items()
        }
        row = {
            "epoch": float(epoch),
            "learning_rate": learning_rate,
            "train_loss_normalized": train_losses["objective"],
            "val_loss_normalized": val_loss,
            "train_forecast_mse_normalized": train_losses["forecast_mse"],
            "train_change_anomaly_mse_normalized": train_losses[
                "change_anomaly_mse"
            ],
            "val_forecast_mse_normalized": val_losses["forecast_mse"],
            "val_change_anomaly_mse_normalized": val_losses["change_anomaly_mse"],
            "is_best_objective": int(improvements["objective"]),
            "is_best_rmse": int(improvements["rmse"]),
            "is_best_anomaly": int(improvements["anomaly"]),
            "is_best_correlation": int(improvements["correlation"]),
            **val_metrics,
        }
        history.append(row)
        save_history(run_dir / "history.csv", history)

        print(
            f"Epoch {epoch:03d} | lr={learning_rate:.2e} | "
            f"train={train_losses['objective']:.6f} | val={val_loss:.6f} | "
            f"val_mse={val_losses['forecast_mse']:.6f} | "
            f"val_anom={val_losses['change_anomaly_mse']:.6f} | "
            f"RMSE={val_metrics['rmse_celsius']:.4f} degC | "
            f"skill={val_metrics['skill_vs_persistence']:.4f}"
        )

        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)
            next_learning_rate = float(optimizer.param_groups[0]["lr"])
            if next_learning_rate < learning_rate:
                print(
                    f"Learning rate reduced: {learning_rate:.2e} -> "
                    f"{next_learning_rate:.2e}"
                )

        for criterion, improved in improvements.items():
            if not improved:
                continue
            best_values[criterion] = criterion_values[criterion]
            best_epochs[criterion] = epoch
            checkpoint = make_checkpoint(
                model,
                args,
                model_config,
                sst_scaler,
                split,
                epoch,
                criterion,
                val_losses,
                val_metrics,
            )
            torch.save(checkpoint, run_dir / f"best_{criterion}.pt")
            if criterion == "objective":
                # Backward-compatible alias used by the existing commands.
                torch.save(checkpoint, run_dir / "best.pt")

        torch.save(
            make_checkpoint(
                model,
                args,
                model_config,
                sst_scaler,
                split,
                epoch,
                "last",
                val_losses,
                val_metrics,
            ),
            run_dir / "last.pt",
        )

        epochs_without_improvement = update_early_stopping_wait(
            epoch,
            args.min_epochs,
            improvements["objective"],
            epochs_without_improvement,
        )
        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after epoch {epoch}: no objective improvement for "
                f"{args.patience} eligible epochs (min_epochs={args.min_epochs})."
            )
            break

    print("Best validation checkpoints:")
    for criterion in criterion_modes:
        print(
            f"  {criterion}: epoch {best_epochs[criterion]}, "
            f"value={best_values[criterion]:.6f}, "
            f"file={run_dir / f'best_{criterion}.pt'}"
        )
    print(f"Backward-compatible objective checkpoint: {run_dir / 'best.pt'}")
    print(f"Final epoch checkpoint: {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
