# eddy-tracking

Research pipeline for tracking mesoscale eddies in the Gulf Stream and overlaying PACE OCI phytoplankton pigment and PFT data.

## Overview

The pipeline ingests daily SWOT L4 SSH data, identifies and tracks mesoscale eddies using [py-eddy-tracker](https://py-eddy-tracker.readthedocs.io/), collocates PACE OCI L3 Rrs spectra to eddy contours, retrieves accessory pigment concentrations via the SDP model (Kramer et al. 2022), and resolves phytoplankton functional types (PFTs) via simulated annealing + NNLS (phytoclass; Hayward et al. 2023).

**Data sources:**
- SWOT L4 SSH: AVISO FTP (`dt_global_allsat_phy_l4_*.nc`)
- PACE OCI L3M Daily RRS: NASA Earthdata (`PACE_OCI_L3M_RRS`, 4 km)
- SST: AQUA MODIS 8-day composites via earthaccess search and OPeNDAP
- SSS: SMAP 8-day running mean via Harmony API

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage its Python environment.
Install uv once with `curl -LsSf https://astral.sh/uv/install.sh | sh`, then from the repo root run:

```bash
uv sync                # build .venv from the locked dependencies (Python 3.10)
cp .env.example .env   # then fill in your AVISO FTP credentials
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

Each experiment has a config directory under `configs/<experiment>/`:

```
configs/
  gulf_stream_20241001_20250701/
    config.yaml     # all stage settings in one file, keyed by section:
                    #   base (region, dates, data paths), eddy_id, eddy_track,
                    #   collocate_pace, phytoclass
    f_matrix.csv    # phytoclass pigment-to-PFT matrix
    min_max.csv     # phytoclass parameter bounds
```

Data is organized in a bronze/silver/gold (medallion) layout: `bronze/` is raw
downloads keyed by dataset, `silver/` is per-experiment processed stages, and
`gold/` is the analysis-ready table.

### Transformation DAG

```text
Bronze downloads
----------------
python -m eddy_tracking.downloads.swot <experiment>
  -> data/{dataset}/bronze/swot_l4/*.nc

python -m eddy_tracking.downloads.pace <experiment>
  -> data/{dataset}/bronze/pace_l3_8d/*.nc

python -m eddy_tracking.downloads.sst <experiment>
  -> data/{dataset}/bronze/sst/*.nc

python -m eddy_tracking.downloads.sss <experiment>
  -> data/{dataset}/bronze/sss/*.nc


Core eddy + pigment branch
--------------------------
data/{dataset}/bronze/swot_l4/*.nc
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


Optional PFT branch
-------------------
silver/pigments/{cyclone,anticyclone}/eddy_*_pigments.parquet
  -> run_phytoclass.py
  -> silver/pft/{cyclone,anticyclone}/eddy_*_pfts.parquet


Gold-table side inputs
----------------------
data/{dataset}/bronze/swot_l4/*.nc
silver/eddy_track/{cyclone,anticyclone}/*_tracks.zarr
  -> gulf_stream.py
  -> silver/gulf_stream/streamline.parquet
  -> silver/gulf_stream/eddy_movement.parquet

data/{dataset}/bronze/swot_l4/*.nc
silver/eddy_track/{cyclone,anticyclone}/*_tracks.zarr
  -> eddy_dynamics.py
  -> silver/eddy_dynamics/{cyclone,anticyclone}/dynamics.parquet

data/{dataset}/bronze/pace_l3_8d/*.nc
data/{dataset}/bronze/swot_l4/*.nc
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
```

`run_phytoclass.py` is a downstream PFT branch from the pigment files. It is
useful for community-composition analyses, but the current gold table does not
join PFT outputs.

### Orchestration

`run_pipeline.py` is a lightweight local subprocess runner for producing the
gold table. It runs the three download stages in parallel and then runs these
stages sequentially:

```text
eddy_id -> eddy_track -> collocate_pace -> run_sdp
  -> gulf_stream -> eddy_dynamics -> background -> build_gold_table
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

`run_phytoclass.py` is available as an explicit optional branch after
`run_sdp.py` when PFT parquet files are needed:

```bash
uv run python run_pipeline.py <experiment> run_phytoclass
```

## HPC (PACE Phoenix cluster)

Slurm scripts are in `slurm/`.
The `submit_pipeline.sh` script is still the older core/PFT cluster runner. It
does not currently submit the gold-table stages
(`gulf_stream`, `eddy_dynamics`, `background`, `build_gold_table`).

```bash
# On the Phoenix login node (GT VPN required):
export EXPERIMENT=<experiment>   # e.g. gulf_stream_20241001_20250701
bash slurm/submit_pipeline.sh "$EXPERIMENT"
```

Each sbatch script requires `$EXPERIMENT` to be set via `sbatch --export` or the submit script.

The `slurm/` scripts activate the uv venv (`source .venv/bin/activate`) and submit with `--account=gts-ldove6 --partition=cpu-small --qos=inferno`.
Before a real run, point `data/` at scratch (`ln -sfn ~/scratch/eddy-data ~/projects/eddy-tracking/data`), since home is only 20 GB, and stage `.env` + `~/.netrc` for the download stages.
See `slurm/README.md` for details.

## Directory structure

```
configs/          per-experiment config.yaml (+ phytoclass CSVs)
data/             per-experiment medallion layers (gitignored, regenerable):
  <experiment>/
    bronze/       raw downloads (SWOT, PACE, SST, SSS)
    silver/       processed stages (eddy_id, eddy_track, collocate_pace,
                  pigments, pft, gulf_stream, eddy_dynamics)
    gold/         analysis-ready eddy-pigment table
outputs/          legacy outputs from older experiments (pre-medallion)
src/eddy_tracking/
  downloads/      importable SWOT, PACE, SST, and SSS download modules
  packages/
    sdp/          SDP pigment model (Kramer et al. 2022)
    phytoclass/   phytoclass PFT decomposition (Hayward et al. 2023)
    py_eddy_tracker/
                  vendored eddy identification and tracking package
utils/
  config.py       config loader, medallion path helpers, METADATA_COLS
notebooks/        exploratory analysis
slurm/            HPC job scripts
todo/hypotheses/  research hypotheses and visualization tasks
archive/          old single-eddy analysis outputs
```
