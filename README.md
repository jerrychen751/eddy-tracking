# eddy-tracking

Research pipeline for Gulf Stream eddy tracks and PACE OCI chlorophyll, pigment, and taxonomic class estimates.

## Research direction

CHL denotes near-surface chlorophyll-a concentration. The current questions concern CHL change with eddy age, radial pigment composition, and evidence for lateral exchange in cyclones that cross the Gulf Stream from north to south and anticyclones that cross it from south to north.

The paragraph and figure plan is in the Obsidian note `Work Vault/Work/1. GT Oceanography Lab/Manuscript Planning.md`. The current pipeline supplies interior pigment estimates and per-pixel pigment retrievals out to `collocate-pace --max-radius` speed radii of each eddy center. The proposed paper also needs local reference water and cohort validation.

## Overview

The pipeline uses daily DUACS-MIOST Level 4 sea surface height (SSH), which combines SWOT and nadir altimetry. [py-eddy-tracker](https://py-eddy-tracker.readthedocs.io/) identifies eddies and links their tracks. The pipeline matches PACE OCI Level 3 remote sensing reflectance (Rrs) to eddy contours. The Spectral Derivative Pigments (SDP) model estimates pigment concentrations (Kramer et al. 2022). The canonical experiment uses eight-day PACE composites.

**Data sources:**
- DUACS-MIOST L4 SSH, including SWOT: AVISO FTP (`dt_global_allsat_phy_l4_*.nc`)
- DUACS near-real-time L4 SSH, including the SWOT swath, for the days after the AVISO record: Copernicus Marine (`cmems_obs-sl_glo_phy-ssh_nrt_demo-allsat-swos-l4-duacs-0.125deg_P1D-i`)
- PACE OCI L3 mapped AOP: NASA Earthdata (`PACE_OCI_L3M_AOP`, 4 km; eight-day composites in the canonical experiment)
- PACE OCI L3 mapped BGC: direct CHL source (`chlor_a`); `collocate_chlorophyll.py` validates inputs and calculates eddy means without SDP
- SST: AQUA MODIS 8-day composites via earthaccess search and OPeNDAP
- SSS: SMAP 8-day running mean via Harmony API
- Copernicus Marine daily L3 plankton, 4 km, multi-sensor GlobColour processing (`cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D`): CHL plus nine phytoplankton group concentrations, each with an uncertainty field; `collocate_plankton.py` builds 8-day eddy means of every concentration and uncertainty field over the whole tracking date range

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage its Python environment.
Install uv once with `curl -LsSf https://astral.sh/uv/install.sh | sh`, then from the repo root run:

```bash
uv sync # build .venv from the locked dependencies (Python 3.13)
cp .env.example .env # then fill in your AVISO FTP and Copernicus Marine credentials
```

`uv sync` reads `pyproject.toml` and `uv.lock` and builds an exact, reproducible environment.
`uv sync` also installs the `eddy-tracking` command. Run it through `uv run`, for example `uv run eddy-tracking identify-eddies --help`, or `source .venv/bin/activate` once and call `eddy-tracking` directly.

py-eddy-tracker (PET) is vendored at `src/eddy_tracking/packages/py_eddy_tracker/`, not installed from PyPI. `eddy_id.py` calls PET's
`grid.eddy_identification(...)` directly and writes explicit output filenames,
so it does not rely on PET's CLI filename template path.

### Phoenix (PACE) cluster

Install uv the same way on Phoenix (its login node can reach PyPI and GitHub), place this repo under `~/projects/`, then run `uv sync --no-dev`.
The `--no-dev` flag skips the Jupyter tooling that batch jobs do not need.
uv downloads its own Python 3.13, so the system Python version on the cluster does not matter.
Keep large data and outputs in the group project space (`/storage/project/r-ldove6-0/<gt_username>`) or scratch, not in the 20 GB home directory.

The `.env` file must contain:
```
FTP_HOST = "ftp-access.aviso.altimetry.fr"
FTP_USER = "<your email>"
FTP_PASSWORD = "<your password>"
```

## Running the tools

Every download and processing step is one command, `eddy-tracking <tool>`, that takes its region, dates, settings, and input and output folders as flags; `eddy-tracking` alone lists the tools and `eddy-tracking <tool> --help` lists a tool's flags. A tool never reads another tool's output on its own: each input is a flag, so any tool can read any study's output. Every scientific setting is a required flag; only worker counts and chunk sizes have defaults.

Each tool writes a `settings.yaml` into its output folder, or `<file>.settings.yaml` beside a single-file output, with the full command, every setting, the git commit (or a hash of the package source on Phoenix, which has no `.git`), and the time. A rerun appends a record. A rerun whose scientific settings differ from an earlier run into the same output stops with exit code 2 and names the differing keys, so two methods never mix in one folder; only `--dates` and the operational settings may change. Notebooks read a tool's settings with `eddy_tracking.config.read_settings(<output>, '<tool>')`. The records of outputs made before the tools existed were written on 2026-10-30 from the config files of that time and say `backfilled`.

A study is a folder under `configs/<study>/` holding `run.sh`, the list of commands that made it, and, for a study that notebooks analyze, `analysis.yaml` with the analysis period:

```
configs/
  gulf_stream_20240305_20260531/            the canonical study
    run.sh
    analysis.yaml
  gulf_stream_20240305_20260531_amplitude_1cm/   1 cm minimum amplitude, reads the canonical bronze data
  kuroshio_20240305_20260531/               the Kuroshio Extension, lon [135, 160] and lat [29, 44]
  bgc_20241001_20250701/                    the SE BGC poster study
  gulf_stream_duacs_19930101_20260515/      reprocessed nadir altimetry 1993 to 2026 matched to Argo profiles
```

Run a study from the repository root with `bash configs/<study>/run.sh`, or copy the lines you need. The downloads, `identify-eddies` and `run-sdp` skip files they already wrote, and `download-argo` resumes at its first missing window, so a rerun continues where they stopped; the other tools redo their whole output.

Data is organized in a bronze/silver/gold (medallion) layout under `data/<study>/`: `bronze/` is raw
downloads keyed by dataset, `silver/` is processed outputs, and
`gold/` is the analysis-ready tables. The folder names are the ones each `run.sh` passes to `--out`.

The AVISO SSH record ends on 2025-12-31. The canonical study continues it to 2026-10-10 with the Copernicus Marine near-real-time maps through `download-cmems-ssh --grid-from`, which reads those maps for the region, interpolates them linearly onto the AVISO grid (the Copernicus cell centers sit half a cell off it), derives `relative_vorticity` from the anomaly velocities, applies the `open_ocean` label of the AVISO files, and writes one file per day into the AVISO folder in the AVISO layout, so every later tool reads both sources alike. Without `--grid-from`, `download-cmems-ssh --exclude-box` keeps the product's own grid and builds its own open-ocean mask, as the 1993 to 2026 study does with the reprocessed maps. It reads the public Copernicus store without a login. Over the 62 days both sources cover (2025-10-28 to 2025-12-31) they differ by 1.6 cm RMS in sea level anomaly, less than the AVISO maps change from one day to the next (2.2 cm), and 97% or more of the AVISO eddies of at least 5 cm amplitude have a Copernicus eddy within one speed radius on the same day.

PACE and SST first request a remote subset. If that request fails, the tool uses Earthaccess to download each remaining global file and subset it locally. The tool deletes each temporary global file before the next download. It skips completed subset files on another run.

`run-sdp --workers` sets the number of SDP processes. The canonical study uses four processes. Each process loads the temperature and salinity grids once. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1` to prevent extra numerical-library threads in each process.

### What each tool reads and writes

For standard PACE chlorophyll, run `collocate-chlorophyll` after `trace-axis`. It requires local 8-day BGC files, which `download-pace --product BGC` fetches from the collection named by `--collection-id`, keeping only files of `--version`.

The canonical study uses 50% valid interior coverage and at least 10 pixels for `collocate-chlorophyll`. The stage checks composite dates, grid coordinates, units, physical track identities, and movement lifetimes before it writes `silver/pace_chl/eddy_chlor_a.parquet`. It preserves the prior table if validation or the write fails. Missing satellite pixels remain valid gaps. The lifetime notebook does not read this table; its chlorophyll source is the plankton table below.

The stage rejects overlapping composite files, including two product versions for one interval. It reports an error when no observations pass the filters. It permits unknown movement classes when an endpoint has no usable Gulf Stream axis. A change to the source files or thresholds requires another stage run.

```text
Bronze downloads
----------------
download-aviso-ssh   AVISO FTP
  -> <out>/dt_global_allsat_phy_l4_<date>_<processing date>.nc, with the open_ocean label and the unmasked <field>_full beside each masked field

download-cmems-ssh   Copernicus Marine store
  -> <out>/<dataset id>_<date>.nc in the AVISO layout, on the --grid-from AVISO grid or the product's own grid

download-pace        NASA Earthdata, one product per run (--product AOP or BGC)
  -> <out>/*.nc

download-sst, download-sss
  -> <out>/*.nc

download-plankton
  -> <out>/plankton_<first day>_<last day>.nc, one per calendar month of --dates

download-glorys
  -> <out>/surface/surface_<first day>_<last day>.nc    mlotst, zos, bottomT (2D), one per calendar month of --dates
  -> <out>/currents/currents_<first day>_<last day>.nc  uo, vo at the top model level
  -> <out>/profiles/profiles_<first day>_<last day>.nc  thetao, so from the surface to --max-depth-m

download-argo        Ifremer ERDDAP through argopy, --profiles core, bgc or both
  -> <out>/core_index.parquet, bgc_index.parquet   GDAC index rows for the profiles inside the box and --dates
  -> <out>/core_profiles.nc                        PRES/TEMP/PSAL, adjusted values preferred; position, time, pressure and temperature good QC, salinity blanked where its QC failed
  -> <out>/bgc_profiles.nc                         all BGC parameters, argopy expert mode (real-time kept; filter on *_QC)

download-insitu
  -> <out>/index/*.txt                 Copernicus in situ TAC index files (bounding box and time span per platform file)
  -> <out>/GL_PR_{GL,CT,XB}_*.nc       the glider, ship CTD and XBT files overlapping the box and --dates, fetched from the public native bucket
  -> <out>/profiles.nc                 those profiles in the Argo layout (N_PROF x N_LEVELS; PLATFORM_NUMBER, PLATFORM_TYPE, DATA_MODE, TIME, LONGITUDE, LATITUDE, PRES, TEMP, PSAL),
                                       good-QC levels only, profiles reaching --min-depth-m, one per platform per --thin-hours

download-woa
  -> <out>/woa23_monthly.nc


Eddies and pigments (the folder names are the canonical study's)
--------------------------------------------------------------
--ssh bronze/swot_l4_open_ocean
  -> identify-eddies -> silver/eddy_id/{cyclone,anticyclone}/*.nc
  -> track-eddies    -> silver/eddy_track/{cyclone,anticyclone}/*_tracks.zarr

--tracks silver/eddy_track, --pace bronze/pace_l3_8d
  -> collocate-pace -> silver/collocate_pace/{cyclone,anticyclone}/eddy_*_rrs.parquet

--collocated silver/collocate_pace, --sst bronze/sst, --sss bronze/sss
  -> run-sdp -> silver/pigments/{cyclone,anticyclone}/eddy_*_pigments.parquet


Axis, dynamics and background
-----------------------------
--ssh, --tracks
  -> trace-axis       -> silver/gulf_stream/streamline.parquet, eddy_movement.parquet
  -> measure-dynamics -> silver/eddy_dynamics/{cyclone,anticyclone}/dynamics.parquet

--ssh, --pace, --sst, --sss, --tracks
  -> sample-background -> silver/pigments/background/bg_mean.parquet


Gold tables
-----------
--pigments silver/pigments, --tracks, --dynamics silver/eddy_dynamics, --axis silver/gulf_stream, --background silver/pigments/background
  -> build-gold-table -> gold/eddy_pigment_table.parquet

--plankton bronze/plankton, --tracks, --axis
  -> collocate-plankton -> gold/eddy_plankton_table.parquet

--tracks, --argo bronze/argo/core_profiles.nc, --axis
  -> match-argo -> gold/argo_matchups.parquet

--argo bronze/argo/core_profiles.nc, --ssh, --axis
  -> build-argo-profiles -> gold/argo_profiles.nc
```

`collocate-plankton` writes one row per eddy and NASA 8-day period, with the mean, mean uncertainty, and pixel count of every plankton field inside the eddy boundary nearest the period midpoint, plus the movement class, physical lifetime, and `age_frac`. A row needs CHL on at least `--min-coverage` of the interior pixels and on at least `--min-pixels` of them. The lifetime notebook reads this table as its chlorophyll source.

`match-argo` writes one row per Argo profile and eddy of the profile's own day whose speed contour holds the profile or whose center lies within `--max-radius-fraction` speed radii of it, with `radius_fraction`, `inside_contour`, the eddy's movement class, `crossing_date` (the first day an eddy that ends on the other side of the axis is seen on that side), `days_from_crossing`, and `profile_axis_km`.

PhytoClass is inactive for now. Its code remains in `src/eddy_tracking/packages/phytoclass/`, but no tool runs PhytoClass or writes PFT outputs.

## HPC (PACE Phoenix cluster)

One Slurm script, `slurm/run.sbatch`, runs whatever command follows it inside the project venv: one tool, or a whole study's `run.sh`.

```bash
# On the Phoenix login node (GT VPN required), from ~/projects/eddy-tracking:
sbatch slurm/run.sbatch eddy-tracking run-sdp --collocated ... --out ...
sbatch --qos=embers --cpus-per-task=24 --mem=64G --time=08:00:00 slurm/run.sbatch bash configs/<study>/run.sh
```

The script submits with `--account=gts-ldove6 --partition=cpu-small --qos=inferno`; `--qos=embers` is free, preemptible, and stops a job at 8 hours.
Before a real run, point `data/` at scratch (`ln -sfn ~/scratch/eddy-data ~/projects/eddy-tracking/data`), since home is only 20 GB, create `logs/`, and stage `.env` + `~/.netrc` for the downloads. PACE and SST downloads cannot run there, because Phoenix cannot reach `oceandata.sci.gsfc.nasa.gov`; run them on the laptop and copy the files up.
See `slurm/README.md` for details.

## Directory structure

```
configs/          per-study run.sh, and analysis.yaml for a study notebooks analyze
data/             per-study medallion layers (gitignored, regenerable):
  <study>/
    bronze/       raw downloads (SWOT, PACE, SST, SSS, Copernicus Marine plankton)
    silver/       processed stages (eddy_id, eddy_track, collocate_pace,
                  pigments, gulf_stream, eddy_dynamics)
    gold/         analysis-ready eddy-pigment and eddy-plankton tables, and the
                  plankton ring table that research_paper/figures.ipynb reads
src/eddy_tracking/
  cli.py          the eddy-tracking command, which maps a tool name to its module
  config.py       analysis.yaml and settings.yaml readers, medallion path helpers for notebooks, METADATA_COLS
  downloads/      the download tools
  tools/          the processing tools
  utils/cli.py    the shared region and date flags and the settings.yaml writer
  preprocess/     readers for the bronze files and the track product, plus the Gulf Stream centerline
  packages/
    sdp/          SDP pigment model (Kramer et al. 2022)
    phytoclass/   retained PhytoClass package; inactive
    py_eddy_tracker/
                  vendored eddy identification and tracking package
scripts/          analysis scripts that read the tool outputs
notebooks/        exploratory notebooks, numbered by kind (see notebooks/README.md), and archive/
research_paper/   paper figure and analysis notebooks
research_updates/ dated advisor update notebooks
conferences/      poster notebooks: space_week.ipynb and se_bgc.ipynb
slurm/            HPC job script that runs one command or one run.sh
```
