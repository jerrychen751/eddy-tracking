# Slurm job scripts

HPC Slurm job scripts for running the pigment branch of the
eddy-tracking pipeline on PACE Phoenix. The local `run_pipeline.py` now runs
the full gold-table path; these Slurm scripts do not yet submit `gulf_stream`,
`eddy_dynamics`, `background`, or `build_gold_table`.
All jobs use `--account=gts-ldove6 --partition=cpu-small --qos=inferno`.
`inferno` is the default charged QOS; switch to `--qos=embers` for free but preemptible backfill.

## Prerequisites

- Place the repo at `~/projects/eddy-tracking/`, then build the env with `uv sync --no-dev`.
- Each script activates the uv virtual environment with `cd ~/projects/eddy-tracking && source .venv/bin/activate` (no conda).
- Point `data/` at scratch, not the 20 GB home dir (the group project space is currently full): `ln -sfn ~/scratch/eddy-data ~/projects/eddy-tracking/data`. Scratch is 15 TB but purges files untouched for 60 days, so copy the small gold parquet to home for long-term keeping.
- Create a `.env` in the project root with `FTP_HOST`, `FTP_USER`, `FTP_PASSWORD`, and ensure `~/.netrc` has Earthdata credentials, before running the download stages.

## How to submit

```bash
# Submit the pigment branch for an experiment
EXPERIMENT=gulf_stream_20241001_20250701 bash slurm/submit_pipeline.sh gulf_stream_20241001_20250701

# Resume from a specific stage (e.g., if eddy_id already ran)
bash slurm/submit_pipeline.sh gulf_stream_20241001_20250701 --from eddy_track
```

`submit_pipeline.sh` submits download stages in parallel (no mutual dependencies), then chains all subsequent pigment stages with `--dependency=afterok` so each stage only starts after the previous one succeeds.

## Stage resource summary

| Stage | CPUs | Memory | Wall time | Notes |
|-------|------|--------|-----------|-------|
| `download_swot` | 4 | 8 GB | 6 h | I/O-bound; FTP parallel downloads |
| `download_pace` | 4 | 16 GB | 12 h | I/O-bound; HTTPS via earthaccess |
| `download_sst_sss` | 4 | 8 GB | 6 h | Harmony API + earthaccess |
| `eddy_id` | 12 | 32 GB | 6 h | CPU-bound; ProcessPoolExecutor |
| `eddy_track` | 4 | 16 GB | 2 h | Single-threaded PET tracking |
| `collocate_pace` | 4 | 32 GB | 4 h | Per-date spatial join |
| `run_sdp` | 8 | 32 GB | 12 h | GSM inversion + pigment ensemble |

## Job dependency chain

```
download_swot ─┐
download_pace ─┼─→ eddy_id → eddy_track → collocate_pace → run_sdp
download_sst_sss ┘
```

The three download stages have no mutual dependency and run simultaneously. The `EXPERIMENT` variable is passed via `--export=ALL,EXPERIMENT=<name>` and validated inside each script with `: "${EXPERIMENT:?...}"`.

## Monitoring

```bash
squeue -u $USER
tail -f logs/<stage>_<jobid>.log
```

## Troubleshooting

- **Job stuck in `PD (dependency)`**: a parent job failed. Check `squeue --jobs <parent_jid>` and the parent's log.
- **OOM kill**: increase `--mem` in the relevant `.sbatch`.
- **earthaccess auth failure**: ensure `~/.netrc` is configured on the compute node (earthaccess writes credentials there after first login).
