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
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from utils.config import PROJECT_ROOT

__all__ = [
    "DEFAULT_STAGES",
    "OPTIONAL_STAGES",
    "PARALLEL_STAGES",
    "VALID_STAGES",
    "main",
    "resolve_stages",
]

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

_STAGE_MODULES = {
    "download_swot": ("eddy_tracking.downloads.swot",),
    "download_pace": ("eddy_tracking.downloads.pace",),
    "download_sst_sss": (
        "eddy_tracking.downloads.sst",
        "eddy_tracking.downloads.sss",
    ),
}


def _log_stage(action: str, stage: str) -> None:
    """Print a timestamped stage status."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] -- {action}: {stage} --")


def _run_stage(experiment: str, stage: str) -> None:
    """Run one stage as a child process, printing start and completion times."""
    modules = _STAGE_MODULES.get(stage)
    if modules is not None:
        commands = [
            [sys.executable, "-m", module, experiment]
            for module in modules
        ]
    else:
        script = PROJECT_ROOT / f"{stage}.py"
        if not script.exists():
            print(
                f"ERROR: Script not found for stage '{stage}': {script}",
                file=sys.stderr,
            )
            sys.exit(1)
        commands = [[sys.executable, str(script), experiment]]

    _log_stage("START", stage)
    for command in commands:
        subprocess.run(command, check=True)
    _log_stage("DONE", stage)


def _run_parallel_downloads(
    experiment: str,
    stages_to_run: Collection[str],
) -> None:
    """Run selected download subprocesses concurrently and exit on failure."""
    download_stages = [
        stage for stage in PARALLEL_STAGES if stage in stages_to_run
    ]
    if not download_stages:
        return

    print(f"Running {len(download_stages)} parallel download(s)...")

    with ThreadPoolExecutor(max_workers=len(download_stages)) as pool:
        futures = {
            pool.submit(_run_stage, experiment, stage): stage
            for stage in download_stages
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


def resolve_stages(
    args_stages: list[str],
    from_stage: str | None,
) -> list[str]:
    """
    Resolve default, ``--from``, or explicitly selected stages.

    Raises ``SystemExit`` when a stage name is unknown.
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


def main() -> None:
    """Parse CLI arguments and run the selected stage subprocesses."""
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

    selected_stages = set(stages_to_run)
    _run_parallel_downloads(args.experiment, selected_stages)

    # Sequential stages (everything after downloads), in canonical order even
    # when the user passes an explicit subset.
    sequential_stages = [
        stage for stage in VALID_STAGES if stage not in PARALLEL_STAGES
    ]
    for stage in sequential_stages:
        if stage in selected_stages:
            _run_stage(args.experiment, stage)

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"Pipeline complete, {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
