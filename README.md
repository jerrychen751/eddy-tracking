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

```bash
# Create the conda environment (environment.yml not yet exported — install manually or ask a collaborator for it)
conda create -n eddy python=3.11
conda activate eddy
# Install key packages: py-eddy-tracker, xarray, zarr, earthaccess, pandas, numpy,
# matplotlib, scipy, cartopy, python-dotenv, scikit-learn, harmony-py

cp .env.example .env   # fill in AVISO FTP credentials
```

> **Note:** To generate a reproducible `environment.yml` from an existing setup: `conda env export --no-builds > environment.yml`

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
  gulf_stream_cyclonic/
    base.yaml       # region, date range, data paths
    eddy_id.yaml    # Bessel filter, contour step, shape error
    eddy_track.yaml # virtual days, min track length, position filter
    collocate_pace.yaml
    phytoclass.yaml
```

Run the full pipeline:
```bash
python run_pipeline.py gulf_stream_cyclonic
```

Run from a specific stage (to resume after a failure):
```bash
python run_pipeline.py gulf_stream_cyclonic --from eddy_id
```

Run specific stages only:
```bash
python run_pipeline.py gulf_stream_cyclonic eddy_id eddy_track
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

## HPC (PACE ICE cluster)

Slurm scripts are in `slurm/`. The `submit_pipeline.sh` script chains all stages via `--dependency=afterok`.

```bash
# On PACE login node:
export EXPERIMENT=gulf_stream_cyclonic
bash slurm/submit_pipeline.sh
```

Each sbatch script requires `$EXPERIMENT` to be set via `sbatch --export` or the submit script.

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
