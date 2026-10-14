# Writes silver/pace_coverage/eddy_coverage.parquet: for each PACE Rrs file of an experiment and each eddy collocate_pace would match to it, the share of the 4 km cells inside the speed contour that hold a complete Rrs spectrum. collocate_pace keeps an eddy only above collocate_pace.min_coverage; this table keeps every eddy, so the matchup count can be read at any threshold.
#
# Usage:
#     python scripts/collocate_pace_coverage.py <experiment>

import sys

import numpy as np
import pandas as pd
import xarray as xr

from eddy_tracking.config import load_config, resolve_data_dir, resolve_output_dir
from eddy_tracking.preprocess.pace import parse_fn_for_date_range
from eddy_tracking.preprocess.tracks import (
    collect_eddies_for_date_range,
    load_date_eddy_index,
    mask_pixels_inside_contour,
)
from eddy_tracking.utils.subset import parse_date_range


def main(experiment: str) -> None:
    cfg = load_config(experiment)
    collocation_cfg = cfg["collocate_pace"]
    date_index = load_date_eddy_index(experiment, date_range=parse_date_range(collocation_cfg["date_range"]))

    rows = []
    for pace_path in sorted(resolve_data_dir(cfg, "pace_dir").glob("*.nc")):
        composite_date_range = parse_fn_for_date_range(pace_path.name, collocation_cfg["temporal_resolution"])
        if composite_date_range is None:
            continue
        representative_date, date_range_start, date_range_end = composite_date_range
        matched_eddies = collect_eddies_for_date_range(date_index, date_range_start, date_range_end)
        if not matched_eddies:
            continue
        with xr.open_dataset(pace_path) as dataset:
            longitudes = dataset["lon"].values
            latitudes = dataset["lat"].values
            finite_spectra = np.isfinite(dataset["Rrs"].values).all(axis=2)  # (n_lat, n_lon, n_wavelength) -> (n_lat, n_lon)

        for eddy in matched_eddies:
            # Testing only the cells of the contour's bounding box gives the same interior cells as the whole grid, at a fraction of the cost.
            in_lon = (longitudes >= eddy.contour_lon.min()) & (longitudes <= eddy.contour_lon.max())
            in_lat = (latitudes >= eddy.contour_lat.min()) & (latitudes <= eddy.contour_lat.max())
            inside_contour = mask_pixels_inside_contour(longitudes[in_lon], latitudes[in_lat], eddy.contour_lon, eddy.contour_lat)
            n_interior_cells = int(inside_contour.sum())
            if n_interior_cells == 0:
                continue
            n_valid_cells = int(finite_spectra[np.ix_(in_lat, in_lon)][inside_contour].sum())
            rows.append({
                "polarity": eddy.polarity,
                "track_id": eddy.track_id,
                "date": pd.Timestamp(representative_date),
                "n_interior_cells": n_interior_cells,
                "n_valid_cells": n_valid_cells,
                "coverage": n_valid_cells / n_interior_cells,
            })

    coverage = pd.DataFrame(rows)
    out_path = resolve_output_dir(experiment, "pace_coverage") / "eddy_coverage.parquet"
    coverage.to_parquet(out_path, index=False)
    print(
        f"output_path: {out_path}\n"
        f"rows_written: {len(coverage)}\n"
        f"rows_at_min_coverage: {int((coverage['coverage'] >= collocation_cfg['min_coverage']).sum())}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
