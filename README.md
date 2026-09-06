# eddy-tracking

Research pipeline for Gulf Stream eddy tracks and PACE OCI chlorophyll, pigment, and taxonomic class estimates.

## Research direction

CHL denotes near-surface chlorophyll-a concentration. The current questions concern CHL change with eddy age, radial pigment composition, and evidence for lateral exchange in southbound cyclones and northbound anticyclones. The [research scope](docs/research_scope.md) defines the hypotheses, source-water distinctions, verified pipeline capability, and evidence requirements.

The paragraph and figure plan is in the Obsidian note `Work Vault/Work/1. GT Oceanography Lab/Manuscript Planning.md`. The current pipeline supplies interior pigment estimates. The proposed paper also needs exterior radial observations, local reference water, and cohort validation.

## Overview

The pipeline uses daily DUACS-MIOST Level 4 sea surface height (SSH), which combines SWOT and nadir altimetry. [py-eddy-tracker](https://py-eddy-tracker.readthedocs.io/) identifies eddies and links their tracks. The pipeline matches PACE OCI Level 3 remote sensing reflectance (Rrs) to eddy contours. The Spectral Derivative Pigments (SDP) model estimates pigment concentrations (Kramer et al. 2022). The canonical experiment uses eight-day PACE composites.

**Data sources:**
- DUACS-MIOST L4 SSH, including SWOT: AVISO FTP (`dt_global_allsat_phy_l4_*.nc`)
- PACE OCI L3 mapped AOP: NASA Earthdata (`PACE_OCI_L3M_AOP`, 4 km; eight-day composites in the canonical experiment)
- PACE OCI L3 mapped BGC: direct CHL source (`chlor_a`); `collocate_chlorophyll.py` validates inputs and calculates eddy means without SDP
- SST: AQUA MODIS 8-day composites via earthaccess search and OPeNDAP
- SSS: SMAP 8-day running mean via Harmony API
- Copernicus Marine daily L3 plankton, 4 km, multi-sensor GlobColour processing (`cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D`): CHL plus nine phytoplankton group concentrations, each with an uncertainty field; `collocate_plankton.py` builds 8-day eddy means of every concentration and uncertainty field over the whole tracking window

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage its Python environment.
Install uv once with `curl -LsSf https://astral.sh/uv/install.sh | sh`, then from the repo root run:

```bash
uv sync # build .venv from the locked dependencies (Python 3.10)
cp .env.example .env # then fill in your AVISO FTP and Copernicus Marine credentials
```

`uv sync` reads `pyproject.toml` and `uv.lock` and builds an exact, reproducible environment.
Run pipeline commands through it with `uv run`, for example `uv run python run_pipeline.py <experiment>`, or `source .venv/bin/activate` once and call `python` directly.

`pyeddytracker` is installed from PyPI. `eddy_id.py` calls PET's
`grid.eddy_identification(...)` directly and writes explicit output filenames,
so it does not rely on PET's CLI filename template path.

### Phoenix (PACE) cluster

Install uv the same way on Phoenix (its login node can reach PyPI and GitHub), place this repo under `~/projects/`, then run `uv sync --no-dev`.
The `--no-dev` flag skips the Jupyter tooling that batch jobs do not need.
uv downloads its own Python 3.10, so the system Python version on the cluster does not matter.
Keep large data and outputs in the group project space (`/storage/project/r-ldove6-0/<gt_username>`) or scratch, not in the 20 GB home directory.

The `.env` file must contain:
```
FTP_HOST = "ftp-access.aviso.altimetry.fr"
FTP_USER = "<your email>"
FTP_PASSWORD = "<your password>"
```

## Running the pipeline

The canonical experiment is `gulf_stream_20240305_20260531`. The [adopted methods](docs/research_scope.md#adopted-experiment-methods) define its ocean mask, speed contour, and annual cycle variables.

Each experiment has a config directory under `configs/<experiment>/`:

```
configs/
  gulf_stream_20240305_20260531/
    config.yaml # all stage settings in one file, keyed by section: base (region, dates, data paths), eddy_id, eddy_track, collocate_pace
```

Data is organized in a bronze/silver/gold (medallion) layout: `bronze/` is raw
downloads keyed by dataset, `silver/` is per-experiment processed stages, and
`gold/` is the analysis-ready table.

The configuration selects `swot_l4_open_ocean` as `swot_dir` for the canonical experiment. Its `bronze/` directory contains all shared inputs directly.

PACE and SST first request a remote subset. If that request fails, the stage uses Earthaccess to download each remaining global file and subset it locally. The stage deletes each temporary global file before the next download. It skips completed subset files on another run.

`run_sdp.max_workers` controls the number of SDP processes and defaults to one. The canonical experiment uses four processes. Each process loads the temperature and salinity grids once. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1` to prevent extra numerical-library threads in each process.

### Transformation DAG

For standard PACE chlorophyll, run `uv run python run_pipeline.py <experiment> collocate_chlorophyll` after `gulf_stream`. This optional stage requires local 8-day BGC files in `base.data.pace_bgc_dir`. The download stage fetches the AOP and BGC collections named in `base.download.pace.collection_ids` and keeps only files of `base.download.pace.version`.

The canonical experiment uses 50% valid interior coverage and at least 10 pixels, set in `collocate_chlorophyll`. The stage checks composite dates, grid coordinates, units, physical track identities, and movement lifetimes before it writes `silver/pace_chl/eddy_chlor_a.parquet`. It preserves the prior table if validation or the write fails. Missing satellite pixels remain valid gaps. The lifetime notebook does not read this table; its chlorophyll source is the plankton table below.

The stage rejects overlapping composite files, including two product versions for one interval. It reports an error when no observations pass the filters. It permits unknown movement classes when an endpoint has no usable Gulf Stream axis. A change to the source files or thresholds requires another stage run.

```text
Bronze downloads
----------------
python -m eddy_tracking.downloads.swot <experiment>
  -> data/{dataset}/bronze/{swot_dir}/*.nc

python -m eddy_tracking.downloads.pace <experiment>
  -> data/{dataset}/bronze/pace_l3_8d/*.nc
  -> data/{dataset}/bronze/pace_l3_8d_bgc/*.nc

python -m eddy_tracking.downloads.sst <experiment>
  -> data/{dataset}/bronze/sst/*.nc

python -m eddy_tracking.downloads.sss <experiment>
  -> data/{dataset}/bronze/sss/*.nc

python -m eddy_tracking.downloads.cmems <experiment>
  -> data/{dataset}/bronze/plankton/plankton_<first day>_<last day>.nc, one per calendar month of eddy_date_range


Core eddy + pigment branch
--------------------------
data/{dataset}/bronze/{swot_dir}/*.nc
  -> eddy_id.py
  -> silver/eddy_id/{cyclone,anticyclone}/*.nc
  -> eddy_track.py
  -> silver/eddy_track/{cyclone,anticyclone}/*_tracks.zarr

silver/eddy_track/{cyclone,anticyclone}/*_tracks.zarr
data/{dataset}/bronze/pace_l3_8d/*.nc
  -> collocate_pace.py
  -> silver/collocate_pace/{cyclone,anticyclone}/eddy_*_rrs.parquet

silver/collocate_pace/{cyclone,anticyclone}/eddy_*_rrs.parquet
data/{dataset}/bronze/sst/*.nc
data/{dataset}/bronze/sss/*.nc
  -> run_sdp.py
  -> silver/pigments/{cyclone,anticyclone}/eddy_*_pigments.parquet


Gold-table side inputs
----------------------
data/{dataset}/bronze/{swot_dir}/*.nc
silver/eddy_track/{cyclone,anticyclone}/*_tracks.zarr
  -> gulf_stream.py
  -> silver/gulf_stream/streamline.parquet
  -> silver/gulf_stream/eddy_movement.parquet

data/{dataset}/bronze/{swot_dir}/*.nc
silver/eddy_track/{cyclone,anticyclone}/*_tracks.zarr
  -> eddy_dynamics.py
  -> silver/eddy_dynamics/{cyclone,anticyclone}/dynamics.parquet

data/{dataset}/bronze/pace_l3_8d/*.nc
data/{dataset}/bronze/{swot_dir}/*.nc
data/{dataset}/bronze/sst/*.nc
data/{dataset}/bronze/sss/*.nc
silver/eddy_track/{cyclone,anticyclone}/*_tracks.zarr
  -> background.py
  -> silver/pigments/background/bg_mean.parquet


Gold table
----------
silver/pigments/{cyclone,anticyclone}/eddy_*_pigments.parquet
silver/eddy_track/{cyclone,anticyclone}/*_tracks.zarr
silver/gulf_stream/streamline.parquet
silver/gulf_stream/eddy_movement.parquet
silver/eddy_dynamics/{cyclone,anticyclone}/dynamics.parquet
silver/pigments/background/bg_mean.parquet
  -> build_gold_table.py
  -> gold/eddy_pigment_table.parquet

data/{dataset}/bronze/plankton/*.nc
silver/eddy_track/{cyclone,anticyclone}/*_tracks.zarr
silver/gulf_stream/eddy_movement.parquet
  -> collocate_plankton.py
  -> gold/eddy_plankton_table.parquet
```

`collocate_plankton.py` writes one row per eddy and NASA 8-day period, with the mean, mean uncertainty, and pixel count of every plankton field inside the eddy boundary nearest the period midpoint, plus the movement class, physical lifetime, and `age_frac`. A row needs CHL on at least `collocate_plankton.min_coverage` of the interior pixels and on at least `min_pixels` of them. The lifetime notebook reads this table as its chlorophyll source.

PhytoClass is inactive for now. Its code remains in `src/eddy_tracking/packages/phytoclass/`, but the pipeline has no PhytoClass stage or PFT outputs.

### Orchestration

`run_pipeline.py` is a lightweight local subprocess runner for producing the
gold table. It runs the four download stages in parallel and then runs these
stages sequentially:

```text
eddy_id -> eddy_track -> collocate_pace -> run_sdp
  -> gulf_stream -> collocate_plankton -> eddy_dynamics -> background -> build_gold_table
```

It is not a full DAG engine: it has a hard-coded stage list, does not infer
dependencies from files, and individual stage scripts own most skip/overwrite
behavior. The hard-coded default stage list is the current gold-table path.

Run the full local gold-table pipeline:

```bash
uv run python run_pipeline.py <experiment>
```

Resume from a stage:

```bash
uv run python run_pipeline.py <experiment> --from run_sdp
```

Run an explicit subset:

```bash
uv run python run_pipeline.py <experiment> eddy_dynamics background build_gold_table
```

## HPC (PACE Phoenix cluster)

Slurm scripts are in `slurm/`.
The `submit_pipeline.sh` script runs through pigment retrieval. It
does not currently submit the gold-table stages
(`gulf_stream`, `eddy_dynamics`, `background`, `build_gold_table`).

```bash
# On the Phoenix login node (GT VPN required):
export EXPERIMENT=<experiment>   # e.g. gulf_stream_20240305_20260531
bash slurm/submit_pipeline.sh "$EXPERIMENT"
```

Each sbatch script requires `$EXPERIMENT` to be set via `sbatch --export` or the submit script.

The `slurm/` scripts activate the uv venv (`source .venv/bin/activate`) and submit with `--account=gts-ldove6 --partition=cpu-small --qos=inferno`.
Before a real run, point `data/` at scratch (`ln -sfn ~/scratch/eddy-data ~/projects/eddy-tracking/data`), since home is only 20 GB, and stage `.env` + `~/.netrc` for the download stages.
See `slurm/README.md` for details.

## Directory structure

```
configs/          per-experiment config.yaml
data/             per-experiment medallion layers (gitignored, regenerable):
  <experiment>/
    bronze/       raw downloads (SWOT, PACE, SST, SSS, Copernicus Marine plankton)
    silver/       processed stages (eddy_id, eddy_track, collocate_pace,
                  pigments, gulf_stream, eddy_dynamics)
    gold/         analysis-ready eddy-pigment and eddy-plankton tables
outputs/          legacy outputs from older experiments (pre-medallion)
src/eddy_tracking/
  downloads/      importable SWOT, PACE, SST, SSS, and Copernicus Marine download modules
  packages/
    sdp/          SDP pigment model (Kramer et al. 2022)
    phytoclass/   retained PhytoClass package; inactive
    py_eddy_tracker/
                  vendored eddy identification and tracking package
utils/
  config.py       config loader, medallion path helpers, METADATA_COLS
notebooks/        exploratory analysis
slurm/            HPC job scripts
docs/             current research scope and evidence requirements
archive/          old single-eddy analysis outputs
```
