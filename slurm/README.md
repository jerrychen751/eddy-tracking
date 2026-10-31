# Slurm job script

One Slurm job script, `run.sbatch`, runs on PACE Phoenix the command you pass after it, inside the project venv: one `eddy-tracking` tool, or a whole study's `run.sh`.
It submits with `--account=gts-ldove6 --partition=cpu-small --qos=inferno`.
`inferno` is the default charged QOS; switch to `--qos=embers` for free but preemptible backfill, which stops a job at 8 hours.

## Prerequisites

- Place the repo at `~/projects/eddy-tracking/`, then build the env with `uv sync --no-dev`, which also installs the `eddy-tracking` command.
- The script activates the uv virtual environment with `cd ~/projects/eddy-tracking && source .venv/bin/activate` (no conda).
- Point `data/` at scratch, not the 20 GB home dir (the group project space is currently full): `ln -sfn ~/scratch/eddy-data ~/projects/eddy-tracking/data`. Scratch is 15 TB but purges files untouched for 60 days, so copy the small gold parquet to home for long-term keeping.
- Create `logs/` in the project root, because Slurm opens the log file before the script runs.
- Create a `.env` in the project root with `FTP_HOST`, `FTP_USER`, `FTP_PASSWORD`, and ensure `~/.netrc` has Earthdata credentials, before running the downloads.

## How to submit

```bash
# A whole study
sbatch --qos=embers --cpus-per-task=24 --mem=64G --time=08:00:00 slurm/run.sbatch bash configs/gulf_stream_duacs_19930101_20260515/run.sh

# One tool, with more cores and a shorter limit than the script's defaults
sbatch --cpus-per-task=16 --mem=96G --time=04:00:00 slurm/run.sbatch eddy-tracking run-sdp --collocated data/<study>/silver/collocate_pace --sst data/<study>/bronze/sst --sss data/<study>/bronze/sss --workers 16 --out data/<study>/silver/pigments
```

A `run.sh` runs its commands one after another and stops at the first that fails. The downloads, `identify-eddies` and `run-sdp` skip the files they already wrote, and `download-argo` resumes at its first missing window, so submitting the same `run.sh` again continues them; the other tools redo their whole output.

## Resources

The script asks for 8 CPUs, 32 GB, and 12 hours. An `sbatch` flag overrides the matching `#SBATCH` line. Request at least as many CPUs as the `--workers` of `identify-eddies` and `run-sdp`; 16 SDP workers need about 6 GB each.

## Monitoring

```bash
squeue -u $USER
tail -f logs/eddy_tracking_<jobid>.log
```

## Troubleshooting

- **OOM kill**: resubmit with a larger `--mem`.
- **earthaccess auth failure**: ensure `~/.netrc` is configured on the compute node (earthaccess writes credentials there after first login).
- **A dependent job stays PENDING with `DependencyNeverSatisfied`**: the cluster does not cancel it when its parent fails, so submit it with `--kill-on-invalid-dep=yes`.
