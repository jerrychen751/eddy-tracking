"""Build the gold table of Copernicus Marine plankton field means inside each tracked eddy, one row per eddy and 8-day composite date range."""

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eddy_tracking.preprocess.tracks import (
    collect_eddies_for_date_range,
    load_date_eddy_index,
    mask_pixels_inside_contour,
)
from eddy_tracking.utils.cli import write_settings
from eddy_tracking.utils.files import write_parquet_atomically


def build_plankton_table(
    plankton_dir: Path,
    track_dir: Path,
    axis_dir: Path,
    first_day: dt.date,
    last_day: dt.date,
    min_coverage: float,
    min_pixels: int,
) -> pd.DataFrame:
    """
    Return one row per tracked eddy and NASA 8-day period from first_day to last_day. Each period starts on day 1 of the year and every eighth day after it, and the last period of a year ends on December 31. A pixel's composite is its mean over the days of the period with data, and each field mean covers the composite pixels inside the eddy boundary nearest the period midpoint that hold that field. The uncertainty of a field averages over the same pixels as the field. A row needs CHL on at least min_coverage of the interior pixels and on at least min_pixels of them; the other fields may have fewer pixels, down to zero, which leaves their mean absent. The flags field only marks land, so it is not aggregated. Concentrations are in mg/m³, uncertainties in percent, and date is the period midpoint.
    """
    date_index = load_date_eddy_index(track_dir, date_range=(first_day, last_day))
    date_ranges = []
    for year in range(first_day.year, last_day.year + 1):
        start = dt.date(year, 1, 1)
        while start.year == year:
            end = min(start + dt.timedelta(days=7), dt.date(year, 12, 31))
            if end >= first_day and start <= last_day:
                date_ranges.append((start, end))
            start = end + dt.timedelta(days=1)
    concentrations = ["CHL", "DIATO", "DINO", "GREEN", "HAPTO", "MICRO", "NANO", "PICO", "PROCHLO", "PROKAR"]
    fields = xr.open_mfdataset(
        sorted(plankton_dir.glob("plankton_*.nc")), combine="by_coords",
    )[concentrations + [f"{field}_uncertainty" for field in concentrations]]
    lon = fields["longitude"].to_numpy()
    lat = fields["latitude"].to_numpy()
    rows = []
    for start, end in date_ranges:
        composite = fields.sel(time=slice(str(start), str(end))).mean("time").load()  # (n_days, n_lat, n_lon) -> (n_lat, n_lon) per field
        for eddy in collect_eddies_for_date_range(date_index, start, end):
            box = composite.isel(
                longitude=np.flatnonzero((lon >= eddy.contour_lon.min()) & (lon <= eddy.contour_lon.max())),
                latitude=np.flatnonzero((lat >= eddy.contour_lat.min()) & (lat <= eddy.contour_lat.max())),
            )
            inside = mask_pixels_inside_contour(box["longitude"].to_numpy(), box["latitude"].to_numpy(), eddy.contour_lon, eddy.contour_lat)
            n_inside = int(inside.sum())
            if n_inside == 0:
                continue
            chl_valid = np.isfinite(box["CHL"].to_numpy()[inside])
            if chl_valid.sum() < min_pixels or chl_valid.mean() < min_coverage:
                continue
            row = {
                "polarity": eddy.polarity, "track_id": eddy.track_id,
                "date": pd.Timestamp(start + (end - start) / 2).as_unit("ns"),
                "center_lon": eddy.center_lon, "center_lat": eddy.center_lat,
                "n_pixels": n_inside, "valid_frac": float(chl_valid.mean()),
            }
            for field in concentrations:
                values = box[field].to_numpy()[inside]
                uncertainty = box[f"{field}_uncertainty"].to_numpy()[inside]
                valid = np.isfinite(values)
                row[field] = float(values[valid].mean()) if valid.any() else np.nan
                row[f"{field}_uncertainty"] = float(np.nanmean(uncertainty[valid])) if valid.any() else np.nan
                row[f"{field}_n_pixels"] = int(valid.sum())
            rows.append(row)
    identity_columns = ["polarity", "track_id"]
    movement = pd.read_parquet(axis_dir / "eddy_movement.parquet")
    table = pd.DataFrame(rows).merge(
        movement[identity_columns + ["movement", "birth_date", "death_date"]], on=identity_columns, how="left",
    )
    lifetime_days = (table["death_date"] - table["birth_date"]).dt.days
    table["age_frac"] = ((table["date"] - table["birth_date"]).dt.days / lifetime_days).clip(0, 1)
    return table


def main(argv: list[str]) -> None:
    """Write the --out table through a temporary file, so a failed build leaves the prior table in place."""
    parser = argparse.ArgumentParser(prog="eddy-tracking collocate-plankton")
    parser.add_argument("--plankton", type=Path, required=True, help="folder of Copernicus Marine plankton files")
    parser.add_argument("--tracks", type=Path, required=True, help="folder track-eddies wrote")
    parser.add_argument("--axis", type=Path, required=True, help="folder trace-axis wrote")
    parser.add_argument("--dates", nargs=2, required=True, metavar=("START", "END"), help="ISO dates, both inclusive; the 8-day periods that overlap them are kept")
    parser.add_argument("--min-coverage", type=float, required=True, help="fraction of interior pixels that must hold CHL")
    parser.add_argument("--min-pixels", type=int, required=True, help="interior pixels that must hold CHL")
    parser.add_argument("--out", type=Path, required=True, help="the .parquet table to write")
    args = parser.parse_args(argv)
    write_settings(args.out, "collocate-plankton", args, operational=set())

    first_day, last_day = [dt.date.fromisoformat(value) for value in args.dates]
    table = build_plankton_table(args.plankton, args.tracks, args.axis, first_day, last_day, args.min_coverage, args.min_pixels)
    write_parquet_atomically(table, args.out)
    print(f"output_file: {args.out}\neddy_composites: {len(table)}")
