"""Combine the example SWOT L3 files along their line dimension."""

import re
from pathlib import Path

import xarray as xr


def get_start_date(filename: str) -> str:
    """Return the sortable start timestamp embedded in a SWOT filename, such as "20230328T234417" from "SWOT_L3_LR_SSH_Basic_474_003_20230328T234417_20230329T003522_v3.0.nc"."""
    date_pattern = r"(\d{8}T\d{6})_\d{8}T\d{6}"
    match = re.search(date_pattern, filename)
    if match:
        return match.group(1)
    return ""


def main() -> None:
    """Combine example inputs and write the merged NetCDF output."""
    script_dir = Path(__file__).resolve().parent
    input_paths = sorted(
        (script_dir / "swot_l3_cycle474" / "raw").glob("*.nc"),
        key=lambda path: get_start_date(path.name),
    )

    datasets = []
    try:
        for input_path in input_paths:
            dataset = xr.open_dataset(input_path)
            dataset = dataset.drop_vars(
                ["i_num_line", "i_num_pixel"], errors="ignore"
            )
            datasets.append(dataset)

        combined = xr.concat(datasets, dim="num_lines")
        for variable_name in combined.variables:
            if combined[variable_name].dtype.kind == "f":
                combined[variable_name].encoding.pop("dtype", None)
                combined[variable_name].encoding.pop("_FillValue", None)

        output_dir = script_dir / "swot_l3_cycle474" / "combined"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "swot_l3_cycle474_combined.nc"
        combined.to_netcdf(output_path)
        print(
            f"output_path: {output_path}\n"
            "status: written"
        )
    finally:
        for dataset in datasets:
            dataset.close()


if __name__ == "__main__":
    main()
