from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from sst_forecasting.data import SSTScaler, build_datasets, load_sst_data
from sst_forecasting.metrics import MetricAccumulator, masked_mse
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


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    amp_scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    loss_mask_mode: str,
) -> float:
    model.train()
    squared_error = 0.0
    selected_count = 0

    for x, y, ocean_mask, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ocean_mask = ocean_mask.to(device, non_blocking=True)
        mask = selected_mask(ocean_mask, loss_mask_mode)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(x)
            loss = masked_mse(prediction, y, mask)
        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()

        count = int(mask.sum().item())
        squared_error += float(loss.item()) * count
        selected_count += count

    return squared_error / max(selected_count, 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    sst_scaler: SSTScaler,
    device: torch.device,
    use_amp: bool,
    loss_mask_mode: str,
) -> tuple[float, dict[str, float | int]]:
    model.eval()
    normalized_squared_error = 0.0
    selected_count = 0
    metrics = MetricAccumulator()

    for x, y, ocean_mask, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ocean_mask = ocean_mask.to(device, non_blocking=True)
        mask = selected_mask(ocean_mask, loss_mask_mode)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(x)
            loss = masked_mse(prediction, y, mask)

        count = int(mask.sum().item())
        normalized_squared_error += float(loss.item()) * count
        selected_count += count

        prediction_real = sst_scaler.inverse_tensor(prediction.float())
        target_real = sst_scaler.inverse_tensor(y.float())
        persistence_real = sst_scaler.inverse_tensor(x[:, -1].float())
        metrics.update(prediction_real, target_real, persistence_real, ocean_mask)

    return normalized_squared_error / max(selected_count, 1), metrics.compute()


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


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("batch-size, epochs, and patience must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay cannot be negative.")
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
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, amp_scaler, device, use_amp, args.loss_mask
        )
        val_loss, val_metrics = validate(
            model, loaders["val"], sst_scaler, device, use_amp, args.loss_mask
        )
        row = {
            "epoch": float(epoch),
            "learning_rate": learning_rate,
            "train_loss_normalized": train_loss,
            "val_loss_normalized": val_loss,
            **val_metrics,
        }
        history.append(row)
        save_history(run_dir / "history.csv", history)

        print(
            f"Epoch {epoch:03d} | lr={learning_rate:.2e} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
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

        if val_loss < best_val_loss - 1e-10:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "model_config": model_config,
                    "scaler": sst_scaler.state_dict(),
                    "split": split,
                    "normalization": args.normalization,
                    "loss_mask": args.loss_mask,
                    "data_path": str(args.data),
                    "start_date": args.start_date,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                run_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    print(f"Best epoch: {best_epoch}; best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
