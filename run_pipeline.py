"""
Pipeline orchestrator for eddy-tracking experiments.

Runs pipeline stages in the correct order with parallel downloads
and stage selection for partial re-runs.

Usage:
    python run_pipeline.py <experiment>                          # gold-table stages
    python run_pipeline.py <experiment> stage1 stage2 ...        # specific stages
    python run_pipeline.py <experiment> --from <stage>           # from stage onward
    python run_pipeline.py <experiment> run_phytoclass           # optional PFT branch
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from utils.config import PROJECT_ROOT

# Default stage order for the analysis-ready gold table.
# Each name matches its script ({stage}.py).
DEFAULT_STAGES = [
    "download_swot",
    "download_pace",
    "download_sst_sss",
    "eddy_id",
    "eddy_track",
    "collocate_pace",
    "run_sdp",
    "gulf_stream",
    "eddy_dynamics",
    "background",
    "build_gold_table",
]

# Optional branch products that are not joined into the current gold table.
OPTIONAL_STAGES = [
    "run_phytoclass",
]

# Canonical run order used for explicit stage lists too.
VALID_STAGES = DEFAULT_STAGES + OPTIONAL_STAGES

# Stages that are independent and can run in parallel
PARALLEL_STAGES = {"download_swot", "download_pace", "download_sst_sss"}


def log_stage(action, stage):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] -- {action}: {stage} --")


def run_stage(experiment, stage):
    """Run a pipeline stage as a subprocess."""
    script = PROJECT_ROOT / f"{stage}.py"
    if not script.exists():
        print(
            f"ERROR: Script not found for stage '{stage}': {script}",
            file=sys.stderr,
        )
        sys.exit(1)

    log_stage("START", stage)
    subprocess.run([sys.executable, str(script), experiment], check=True)
    log_stage("DONE", stage)


def run_parallel_downloads(experiment, stages_to_run):
    """
    Each download is a subprocess, so threads just wait on I/O —
    the GIL doesn't matter here. Exits on first failure.
    """
    downloads = [s for s in PARALLEL_STAGES if s in stages_to_run]
    if not downloads:
        return

    print(f"Running {len(downloads)} parallel download(s)...")

    with ThreadPoolExecutor(max_workers=len(downloads)) as pool:
        futures = {
            pool.submit(run_stage, experiment, stage): stage
            for stage in downloads
        }
        for future in as_completed(futures):
            stage = futures[future]
            try:
                future.result()
            except subprocess.CalledProcessError:
                print(
                    f"ERROR: Download stage '{stage}' failed.",
                    file=sys.stderr,
                )
                sys.exit(1)

    print("All downloads complete.")


def resolve_stages(args_stages, from_stage):
    """
    Three modes:
        1. No stages and no --from → default gold-table stages
        2. --from <stage> → that stage and everything after it in its branch
        3. Explicit stage list → only those stages
    """
    if from_stage:
        if from_stage not in VALID_STAGES:
            print(
                f"ERROR: Unknown stage '{from_stage}'.\n"
                f"Valid stages: {', '.join(VALID_STAGES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        if from_stage in DEFAULT_STAGES:
            start_idx = DEFAULT_STAGES.index(from_stage)
            return DEFAULT_STAGES[start_idx:]
        return [from_stage]

    if not args_stages:
        return list(DEFAULT_STAGES)

    for stage in args_stages:
        if stage not in VALID_STAGES:
            print(
                f"ERROR: Unknown stage '{stage}'.\n"
                f"Valid stages: {', '.join(VALID_STAGES)}",
                file=sys.stderr,
            )
            sys.exit(1)
    return args_stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("stages", nargs="*")
    parser.add_argument("--from", dest="from_stage", metavar="STAGE")
    args = parser.parse_args()

    stages_to_run = resolve_stages(args.stages, args.from_stage)

    config_dir = PROJECT_ROOT / "configs" / args.experiment
    if not config_dir.is_dir():
        print(
            f"ERROR: Config directory not found: {config_dir}\n"
            f"Create configs/{args.experiment}/ with config.yaml before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    start_time = time.monotonic()
    print(f"Pipeline: {args.experiment}")
    print(f"Stages: {' '.join(stages_to_run)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Parallel downloads
    stages_set = set(stages_to_run)
    run_parallel_downloads(args.experiment, stages_set)

    # Sequential stages (everything after downloads), in canonical order even
    # when the user passes an explicit subset.
    sequential = [s for s in VALID_STAGES if s not in PARALLEL_STAGES]
    for stage in sequential:
        if stage in stages_set:
            run_stage(args.experiment, stage)

    elapsed = time.monotonic() - start_time
    mins, secs = divmod(int(elapsed), 60)
    print(f"Pipeline complete, {mins}m {secs}s")


if __name__ == "__main__":
    main()
