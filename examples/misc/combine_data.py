import xarray as xr
from glob import glob
from pathlib import Path
import re
import os

SCRIPT_DIR = Path(__file__).resolve().parent

def get_start_date(fn: str) -> str:
    date_pattern = r'(\d{8}T\d{6})_\d{8}T\d{6}'
    match = re.search(date_pattern, fn)
    if match:
        return match.group(1)
    return ''

# Collect and sort files by start date
files = glob(str(SCRIPT_DIR / 'swot_l3_cycle474/raw/*.nc'))
files.sort(key=get_start_date)

# Open all datasets, drop incompatible num_nadir variables, concatenate along num_lines
datasets = []
for f in files:
    ds = xr.open_dataset(f)
    # Drop variables with num_nadir dimension (varies between files)
    ds = ds.drop_vars(['i_num_line', 'i_num_pixel'], errors='ignore')
    datasets.append(ds)

combined = xr.concat(datasets, dim='num_lines')

output_dir = str(SCRIPT_DIR / 'swot_l3_cycle474/combined')
os.makedirs(output_dir, exist_ok=True)

# Clear integer encoding for float variables to avoid NaN issues
for var in combined.variables:
    if combined[var].dtype.kind == 'f':  # float variables
        combined[var].encoding.pop('dtype', None)
        combined[var].encoding.pop('_FillValue', None)

# Write merged dataset
output_path = os.path.join(output_dir, 'swot_l3_cycle474_combined.nc')
combined.to_netcdf(output_path)
print(f"Written combined dataset to: {output_path}")

# Clean up
for ds in datasets:
    ds.close()