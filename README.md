# eddy-tracking

Research pipeline for tracking mesoscale eddies in the Gulf Stream and overlaying PACE OCI phytoplankton pigment and PFT data.

## Overview

The pipeline ingests daily SWOT L4 SSH data, identifies and tracks mesoscale eddies using [py-eddy-tracker](https://py-eddy-tracker.readthedocs.io/), collocates PACE OCI L3 Rrs spectra to eddy contours, retrieves accessory pigment concentrations via the SDP model (Kramer et al. 2022), and resolves phytoplankton functional types (PFTs) via simulated annealing + NNLS (phytoclass; Hayward et al. 2023).

**Data sources:**
- SWOT L4 SSH: AVISO FTP (`dt_global_allsat_phy_l4_*.nc`)
- PACE OCI L3M Daily RRS: NASA Earthdata (`PACE_OCI_L3M_RRS`, 4 km)
- SST: AQUA MODIS 8-day composites via earthaccess
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
    base.yaml       # region, date range, data paths
    eddy_id.yaml    # Bessel filter, contour step, shape error
    eddy_track.yaml # virtual days, min track length, position filter
    collocate_pace.yaml
    phytoclass.yaml
```

Run the full pipeline:
```bash
python run_pipeline.py gulf_stream_20241001_20250701
```

Run from a specific stage (to resume after a failure):
```bash
python run_pipeline.py gulf_stream_20241001_20250701 --from eddy_id
```

Run specific stages only:
```bash
python run_pipeline.py gulf_stream_20241001_20250701 eddy_id eddy_track
```

**Stage order:**
1. `download_swot` — AVISO FTP → `data/<dataset>/swot_l4/`
2. `download_pace` — Earthdata HTTPS → `data/<dataset>/pace_l3/`
3. `download_sst_sss` — earthaccess + Harmony → `data/<dataset>/sst/`, `sss/`
4. `eddy_id` — py-eddy-tracker identification → `outputs/<experiment>/eddy_id/`
5. `eddy_track` — py-eddy-tracker tracking → `outputs/<experiment>/eddy_track/`
6. `collocate_pace` — PACE Rrs inside eddy contours → `outputs/<experiment>/collocate_pace/`
7. `run_sdp` — SDP pigment inversion → `outputs/<experiment>/pigments/`
8. `run_phytoclass` — PFT decomposition → `outputs/<experiment>/pft/`

Stages 1–3 run in parallel; stages 4–8 are sequential.

## HPC (PACE Phoenix cluster)

Slurm scripts are in `slurm/`.
The `submit_pipeline.sh` script chains all stages via `--dependency=afterok`.

```bash
# On the Phoenix login node (GT VPN required):
export EXPERIMENT=<experiment>   # e.g. gulf_stream_20241001_20250701
bash slurm/submit_pipeline.sh
```

Each sbatch script requires `$EXPERIMENT` to be set via `sbatch --export` or the submit script.

The `slurm/` scripts activate the uv venv (`source .venv/bin/activate`) and submit with `--account=gts-ldove6 --partition=cpu-small --qos=inferno`.
Before a real run, point `data/` at scratch (`ln -sfn ~/scratch/eddy-data ~/projects/eddy-tracking/data`), since home is only 20 GB, and stage `.env` + `~/.netrc` for the download stages.
See `slurm/README.md` for details.

## Directory structure

```
configs/          per-experiment YAML configs
data/             input data (gitignored, too large)
outputs/          pipeline outputs (gitignored, regenerable)
utils/
  config.py       YAML loader, path helpers, METADATA_COLS
  sdp/            SDP pigment model (Kramer et al. 2022)
  phytoclass/     phytoclass PFT decomposition (Hayward et al. 2023)
notebooks/        exploratory analysis
slurm/            HPC job scripts
todo/hypotheses/  research hypotheses and visualization tasks
archive/          old single-eddy analysis outputs
```
