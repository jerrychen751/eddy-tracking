# Slurm job script

One Slurm job script, `pipeline.sbatch`, runs `python -m eddy_tracking.pipeline` on PACE Phoenix with the arguments you pass to `sbatch`, so it covers every stage the local runner covers, from the downloads to `build_gold_table`.
It submits with `--account=gts-ldove6 --partition=cpu-small --qos=inferno`.
`inferno` is the default charged QOS; switch to `--qos=embers` for free but preemptible backfill.

## Prerequisites

- Place the repo at `~/projects/eddy-tracking/`, then build the env with `uv sync --no-dev`.
- The script activates the uv virtual environment with `cd ~/projects/eddy-tracking && source .venv/bin/activate` (no conda).
- Point `data/` at scratch, not the 20 GB home dir (the group project space is currently full): `ln -sfn ~/scratch/eddy-data ~/projects/eddy-tracking/data`. Scratch is 15 TB but purges files untouched for 60 days, so copy the small gold parquet to home for long-term keeping.
- Create `logs/` in the project root, because Slurm opens the log file before the script runs.
- Create a `.env` in the project root with `FTP_HOST`, `FTP_USER`, `FTP_PASSWORD`, and ensure `~/.netrc` has Earthdata credentials, before running the download stages.

## How to submit

```bash
# Every stage for an experiment
sbatch slurm/pipeline.sbatch gulf_stream_20240305_20260531

# Resume from a specific stage (e.g., if eddy_id already ran)
sbatch slurm/pipeline.sbatch gulf_stream_20240305_20260531 --from eddy_track

# An explicit subset, with more cores and a shorter limit than the script's defaults
sbatch --cpus-per-task=16 --time=04:00:00 slurm/pipeline.sbatch gulf_stream_20240305_20260531 run_sdp
```

The stages run one after another inside the job, in the order of the local runner, and the job stops at the first stage that fails.

## Resources

The script asks for 8 CPUs, 32 GB, and 12 hours, which covers `run_sdp`, the slowest stage. An `sbatch` flag overrides the matching `#SBATCH` line. The parallel stages take their worker count from the `max_workers` keys of the experiment config (`eddy_id` and `run_sdp`), so request at least that many CPUs.

## Monitoring

```bash
squeue -u $USER
tail -f logs/pipeline_<jobid>.log
```

## Troubleshooting

- **OOM kill**: resubmit with a larger `--mem`.
- **earthaccess auth failure**: ensure `~/.netrc` is configured on the compute node (earthaccess writes credentials there after first login).
