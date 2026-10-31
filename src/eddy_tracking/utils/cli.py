# Flags every eddy-tracking tool shares, and the settings record each tool writes beside its output

import argparse
import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from eddy_tracking.config import PROJECT_ROOT


def add_region_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lon", nargs=2, type=float, required=True, metavar=("WEST", "EAST"), help="degrees east")
    parser.add_argument("--lat", nargs=2, type=float, required=True, metavar=("SOUTH", "NORTH"), help="degrees north")
    parser.add_argument("--dates", nargs=2, required=True, metavar=("START", "END"), help="ISO dates, both inclusive")


# Append this run to the output's settings.yaml, or to <file>.settings.yaml beside a single-file output. A rerun may add days or change an operational setting such as a worker count; any other difference would mix two methods in one output, so the tool stops before it writes anything.
def write_settings(out: Path, tool: str, args: argparse.Namespace, operational: set[str]) -> None:
    record_path = out.with_name(f"{out.name}.settings.yaml") if out.suffix else out / "settings.yaml"
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    runs = yaml.safe_load(record_path.read_text(encoding="utf-8")) if record_path.exists() else []

    free = operational | {"dates"}
    # Two tools can write into one folder (download-cmems-ssh continues the AVISO record in its folder), so compare runs of this tool only
    for run in (run for run in runs if run["tool"] == tool):
        differing = sorted(
            key for key in settings.keys() | run["settings"].keys()
            if key not in free and settings.get(key) != run["settings"].get(key)
        )
        if differing:
            print(
                "status: error\n"
                "reason: settings_differ_from_earlier_run\n"
                f"settings_file: {record_path}\n"
                f"differing_keys: {', '.join(differing)}\n"
                "fix: write to another --out folder, or delete the old output first",
                file=sys.stderr,
            )
            sys.exit(2)

    git = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--", "src"], capture_output=True, text=True)
    # Phoenix receives the code by rsync without .git, so a hash of the package source identifies the code there too
    source = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "src" / "eddy_tracking").rglob("*.py")):
        source.update(path.read_bytes())
    runs.append({
        "tool": tool,
        "ran_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if git.returncode == 0 else None,
        "source_sha256": source.hexdigest()[:16],
        "command": " ".join(["eddy-tracking", tool, *sys.argv[2:]]),
        "settings": settings,
    })

    record_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = record_path.with_suffix(".tmp")
    temporary_path.write_text(yaml.safe_dump(runs, sort_keys=False), encoding="utf-8")
    temporary_path.replace(record_path)
