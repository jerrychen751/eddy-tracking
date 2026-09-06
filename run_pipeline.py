"""
Pipeline orchestrator for eddy-tracking experiments.

Runs pipeline stages in the correct order with parallel downloads and stage selection for partial re-runs.

Usage:
    python run_pipeline.py <experiment> # gold-table stages
    python run_pipeline.py <experiment> stage1 stage2 ... # specific stages
    python run_pipeline.py <experiment> --from <stage> # from stage onward
"""

import argparse
import subprocess
import sys
import time
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from utils.config import PROJECT_ROOT

# A stage name resolves to a module inside _run_stage, or else to <stage>.py at the project root: eddy_id -> eddy_id.py.
DEFAULT_STAGES = [
    "download_swot",
    "download_pace",
    "download_sst_sss",
    "download_cmems",
    "eddy_id",
    "eddy_track",
    "collocate_pace",
    "run_sdp",
    "gulf_stream",
    "collocate_plankton",
    "eddy_dynamics",
    "background",
    "build_gold_table",
]

# Canonical run order used for explicit stage lists too.
VALID_STAGES = DEFAULT_STAGES.copy()
VALID_STAGES.insert(VALID_STAGES.index("gulf_stream") + 1, "collocate_chlorophyll")

PARALLEL_STAGES = {"download_swot", "download_pace", "download_sst_sss", "download_cmems"}


def _log_stage(action: str, stage: str) -> None:
    """Print a timestamped stage status."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"timestamp: {timestamp}\n"
        f"action: {action}\n"
        f"stage: {stage}"
    )


def _run_stage(experiment: str, stage: str) -> None:
    """Run one stage as a child process, printing start and completion times."""
    # A download stage runs as a module because it lives under src/, not as a <stage>.py script at the project root.
    stage_modules = {
        "download_swot": ("eddy_tracking.downloads.swot",),
        "download_pace": ("eddy_tracking.downloads.pace",),
        "download_sst_sss": (
            "eddy_tracking.downloads.sst",
            "eddy_tracking.downloads.sss",
        ),
        "download_cmems": ("eddy_tracking.downloads.cmems",),
    }
    modules = stage_modules.get(stage)
    if modules is not None:
        commands = [
            [sys.executable, "-m", module, experiment]
            for module in modules
        ]
    else:
        script = PROJECT_ROOT / f"{stage}.py"
        if not script.exists():
            print(
                "status: error\n"
                "reason: script_not_found\n"
                f"stage: {stage}\n"
                f"script: {script}",
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

    print(
        "status: running_parallel_downloads\n"
        f"downloads: {len(download_stages)}"
    )

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
                    "status: error\n"
                    "reason: download_stage_failed\n"
                    f"stage: {stage}",
                    file=sys.stderr,
                )
                sys.exit(1)

    print("status: downloads_complete")


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
                "status: error\n"
                "reason: unknown_stage\n"
                f"stage: {from_stage}\n"
                f"valid_stages: {', '.join(VALID_STAGES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        start_idx = VALID_STAGES.index(from_stage)
        return [
            stage for stage in VALID_STAGES[start_idx:]
            if stage in DEFAULT_STAGES or stage == from_stage
        ]

    if not args_stages:
        return list(DEFAULT_STAGES)

    for stage in args_stages:
        if stage not in VALID_STAGES:
            print(
                "status: error\n"
                "reason: unknown_stage\n"
                f"stage: {stage}\n"
                f"valid_stages: {', '.join(VALID_STAGES)}",
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
            "status: error\n"
            "reason: config_directory_not_found\n"
            f"config_dir: {config_dir}\n"
            f"experiment: {args.experiment}\n"
            f"required_config_file: {config_dir / 'config.yaml'}",
            file=sys.stderr,
        )
        sys.exit(1)

    start_time = time.monotonic()
    print(f"pipeline: {args.experiment}")
    print(f"stages: {' '.join(stages_to_run)}")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    selected_stages = set(stages_to_run)
    _run_parallel_downloads(args.experiment, selected_stages)

    # Run in canonical order even when the user passes an explicit subset in another order.
    sequential_stages = [
        stage for stage in VALID_STAGES if stage not in PARALLEL_STAGES
    ]
    for stage in sequential_stages:
        if stage in selected_stages:
            _run_stage(args.experiment, stage)

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print(
        "status: pipeline_complete\n"
        f"elapsed_minutes: {minutes}\n"
        f"elapsed_seconds: {seconds}"
    )


if __name__ == "__main__":
    main()
