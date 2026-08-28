from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SSTData:
    """Daily SST fields and optional coordinate metadata."""

    sst: np.ndarray
    dates: np.ndarray
    lon: np.ndarray | None = None
    lat: np.ndarray | None = None


def _canonicalize_sst(array: np.ndarray) -> np.ndarray:
    """Convert common legacy layouts to [day, height, width]."""

    array = np.asarray(array)
    if array.ndim == 5 and array.shape[1] == 1 and array.shape[2] == 1:
        array = array[:, 0, 0]
    elif array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(
            "SST data must have shape [D,H,W], [D,1,H,W], or [D,1,1,H,W]; "
            f"received {array.shape}."
        )
    return array.astype(np.float32, copy=False)


def _make_dates(n_days: int, start_date: str | None) -> np.ndarray:
    if start_date is None:
        return np.asarray([f"day_{index:04d}" for index in range(n_days)])
    start = np.datetime64(start_date, "D")
    offsets = np.arange(n_days).astype("timedelta64[D]")
    return (start + offsets).astype("datetime64[D]")


def load_sst_data(path: str | Path, start_date: str | None = None) -> SSTData:
    """Load a legacy NPY file or a prepared NPZ file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    if path.suffix.lower() == ".npy":
        sst = _canonicalize_sst(np.load(path))
        return SSTData(sst=sst, dates=_make_dates(len(sst), start_date))

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "sst" not in archive:
                raise KeyError(f"{path} does not contain an 'sst' array.")
            sst = _canonicalize_sst(archive["sst"])
            dates = archive["dates"] if "dates" in archive else _make_dates(len(sst), start_date)
            lon = archive["lon"] if "lon" in archive else None
            lat = archive["lat"] if "lat" in archive else None
        if len(dates) != len(sst):
            raise ValueError(f"Date count {len(dates)} does not match SST day count {len(sst)}.")
        return SSTData(sst=sst, dates=dates, lon=lon, lat=lat)

    raise ValueError(f"Unsupported data format '{path.suffix}'. Use .npy or .npz.")


class SSTScaler:
    """Train-only global scaler that preserves NaN values."""

    METHODS = {"none", "minmax", "zscore"}

    def __init__(self, method: str = "minmax") -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown normalization '{method}'. Choose from {sorted(self.METHODS)}.")
        self.method = method
        self.offset = 0.0
        self.scale = 1.0
        self.fitted = False

    def fit(self, train_data: np.ndarray) -> "SSTScaler":
        valid = np.asarray(train_data, dtype=np.float32)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            raise ValueError("Training data contain no finite ocean values.")

        if self.method == "minmax":
            self.offset = float(valid.min())
            self.scale = float(valid.max() - valid.min())
        elif self.method == "zscore":
            self.offset = float(valid.mean())
            self.scale = float(valid.std())
        else:
            self.offset = 0.0
            self.scale = 1.0

        if not np.isfinite(self.scale) or self.scale < 1e-8:
            raise ValueError(f"Invalid scale {self.scale}; training values may be constant.")
        self.fitted = True
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return ((np.asarray(array, dtype=np.float32) - self.offset) / self.scale).astype(
            np.float32, copy=False
        )

    def inverse_numpy(self, array: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(array) * self.scale + self.offset

    def inverse_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        return tensor * self.scale + self.offset

    def state_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {"method": self.method, "offset": self.offset, "scale": self.scale}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SSTScaler":
        scaler = cls(str(state["method"]))
        scaler.offset = float(state["offset"])
        scaler.scale = float(state["scale"])
        scaler.fitted = True
        return scaler

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("SSTScaler.fit must be called before transform or inverse_transform.")


class SSTWindowDataset(Dataset):
    """Sliding windows whose target dates belong to one explicit split."""

    def __init__(
        self,
        scaled_sst: np.ndarray,
        seq_len: int,
        target_start: int,
        target_end: int,
    ) -> None:
        if scaled_sst.ndim != 3:
            raise ValueError(f"Expected [D,H,W], received {scaled_sst.shape}.")
        if seq_len < 1:
            raise ValueError("seq_len must be positive.")
        if target_start < seq_len:
            raise ValueError("The first target must have seq_len preceding input days.")
        if target_end > len(scaled_sst) or target_start >= target_end:
            raise ValueError("Invalid target range.")

        self.sst = scaled_sst
        self.seq_len = seq_len
        self.target_indices = np.arange(target_start, target_end, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        target_index = int(self.target_indices[index])
        x = self.sst[target_index - self.seq_len : target_index]
        y = self.sst[target_index]
        ocean_mask = np.isfinite(y)

        # Fill after normalization: 0 means minimum (minmax), mean (zscore), or 0°C (none).
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.from_numpy(x[:, None].copy()),
            torch.from_numpy(y[None].copy()),
            torch.from_numpy(ocean_mask[None].copy()),
            torch.tensor(target_index, dtype=torch.long),
        )


def build_datasets(
    sst: np.ndarray,
    seq_len: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    normalization: str = "minmax",
) -> tuple[dict[str, SSTWindowDataset], SSTScaler, dict[str, int | float]]:
    """Split by target date and fit normalization on training days only."""

    sst = _canonicalize_sst(sst)
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.")

    n_days = len(sst)
    train_end = int(n_days * train_ratio)
    val_end = int(n_days * (train_ratio + val_ratio))
    if train_end <= seq_len or val_end <= train_end or val_end >= n_days:
        raise ValueError("Not enough days for the requested sequence length and split ratios.")

    scaler = SSTScaler(normalization).fit(sst[:train_end])
    scaled_sst = scaler.transform(sst)
    datasets = {
        "train": SSTWindowDataset(scaled_sst, seq_len, seq_len, train_end),
        "val": SSTWindowDataset(scaled_sst, seq_len, train_end, val_end),
        "test": SSTWindowDataset(scaled_sst, seq_len, val_end, n_days),
    }
    split = {
        "n_days": n_days,
        "seq_len": seq_len,
        "train_end": train_end,
        "val_end": val_end,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
    }
    return datasets, scaler, split
