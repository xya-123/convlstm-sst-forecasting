from __future__ import annotations

import argparse
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np


DATE_PATTERN = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def parse_date_from_name(path: Path) -> date:
    match = DATE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot find an 8-digit date in filename: {path.name}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def collect_daily_files(input_dir: Path, allow_gaps: bool) -> list[tuple[date, Path]]:
    files = [(parse_date_from_name(path), path) for path in input_dir.rglob("*.nc")]
    if not files:
        raise FileNotFoundError(f"No .nc files found below {input_dir}")
    files.sort(key=lambda item: item[0])

    dates = [item[0] for item in files]
    duplicates = sorted({value for value in dates if dates.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate dates found: {duplicates[:10]}")

    expected = []
    current = dates[0]
    while current <= dates[-1]:
        expected.append(current)
        current += timedelta(days=1)
    missing = sorted(set(expected) - set(dates))
    if missing and not allow_gaps:
        preview = ", ".join(value.isoformat() for value in missing[:10])
        raise ValueError(f"Missing {len(missing)} daily files, including: {preview}")
    if missing:
        print(f"Warning: allowing {len(missing)} missing dates.")
    return files


def extract_region(
    path: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import xarray as xr
    except ImportError as error:
        raise RuntimeError(
            "xarray and netCDF4 are required for NetCDF preprocessing. "
            "Install them with: pip install -r requirements.txt"
        ) from error

    with xr.open_dataset(path) as dataset:
        for variable in ("sst", "lon", "lat"):
            if variable not in dataset:
                raise KeyError(f"{path} does not contain required variable '{variable}'.")

        sst = dataset["sst"].squeeze(drop=True)
        if "lat" not in sst.dims or "lon" not in sst.dims:
            raise ValueError(f"Unexpected SST dimensions in {path}: {sst.dims}")
        sst = sst.where(
            (dataset["lon"] >= lon_min)
            & (dataset["lon"] <= lon_max)
            & (dataset["lat"] >= lat_min)
            & (dataset["lat"] <= lat_max),
            drop=True,
        ).transpose("lat", "lon")

        values = sst.values.astype(np.float32)
        lon = sst["lon"].values.astype(np.float32)
        lat = sst["lat"].values.astype(np.float32)
    return values, lon, lat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chronological regional OISST data.")
    parser.add_argument("--input-dir", type=Path, default=Path("2020"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/sst_2020_east_china_sea.npz"),
    )
    parser.add_argument("--lon-min", type=float, default=100.0)
    parser.add_argument("--lon-max", type=float, default=132.0)
    parser.add_argument("--lat-min", type=float, default=16.0)
    parser.add_argument("--lat-max", type=float, default=48.0)
    parser.add_argument("--allow-gaps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lon_min >= args.lon_max or args.lat_min >= args.lat_max:
        raise ValueError("Minimum longitude/latitude must be smaller than the maximum.")

    files = collect_daily_files(args.input_dir, args.allow_gaps)
    fields: list[np.ndarray] = []
    reference_lon: np.ndarray | None = None
    reference_lat: np.ndarray | None = None

    for index, (day, path) in enumerate(files, start=1):
        field, lon, lat = extract_region(
            path, args.lon_min, args.lon_max, args.lat_min, args.lat_max
        )
        if reference_lon is None:
            reference_lon, reference_lat = lon, lat
        elif not np.array_equal(lon, reference_lon) or not np.array_equal(lat, reference_lat):
            raise ValueError(f"Coordinate grid changed in {path}")
        fields.append(field)
        print(f"[{index:03d}/{len(files):03d}] {day.isoformat()} {path.name}")

    sst = np.stack(fields).astype(np.float32)
    dates = np.asarray([day.isoformat() for day, _ in files], dtype="U10")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sst=sst,
        dates=dates,
        lon=reference_lon,
        lat=reference_lat,
    )

    valid = np.isfinite(sst)
    print(f"Saved: {args.output}")
    print(f"Shape: {sst.shape} [day, lat, lon]")
    print(f"Dates: {dates[0]} to {dates[-1]}")
    print(f"Finite ocean fraction: {valid.mean():.4f}")
    print(f"Valid SST range: {np.nanmin(sst):.3f} to {np.nanmax(sst):.3f} °C")


if __name__ == "__main__":
    main()
