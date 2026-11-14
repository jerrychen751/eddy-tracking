# The eddy-tracking command: `eddy-tracking <tool> [flags]` runs main(argv) of the module that owns the tool, and `eddy-tracking <tool> --help` lists its flags

import importlib
import sys

TOOLS = {
    "download-aviso-ssh": "eddy_tracking.downloads.swot",
    "download-cmems-ssh": "eddy_tracking.downloads.cmems_ssh",
    "download-pace": "eddy_tracking.downloads.pace",
    "download-sst": "eddy_tracking.downloads.sst",
    "download-sss": "eddy_tracking.downloads.sss",
    "download-plankton": "eddy_tracking.downloads.cmems",
    "download-glorys": "eddy_tracking.downloads.glorys",
    "download-argo": "eddy_tracking.downloads.argo",
    "download-insitu": "eddy_tracking.downloads.insitu",
    "download-woa": "eddy_tracking.downloads.woa",
    "identify-eddies": "eddy_tracking.tools.eddy_id",
    "track-eddies": "eddy_tracking.tools.eddy_track",
    "trace-axis": "eddy_tracking.tools.gulf_stream",
    "collocate-pace": "eddy_tracking.tools.collocate_pace",
    "run-sdp": "eddy_tracking.tools.run_sdp",
    "collocate-plankton": "eddy_tracking.tools.collocate_plankton",
    "collocate-chlorophyll": "eddy_tracking.tools.collocate_chlorophyll",
    "measure-dynamics": "eddy_tracking.tools.eddy_dynamics",
    "sample-background": "eddy_tracking.tools.background",
    "build-gold-table": "eddy_tracking.tools.build_gold_table",
    "match-argo": "eddy_tracking.tools.match_argo",
    "build-argo-profiles": "eddy_tracking.tools.build_argo_profiles",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in TOOLS:
        print("usage: eddy-tracking <tool> [flags]\ntools:\n  " + "\n  ".join(TOOLS), file=sys.stderr)
        sys.exit(2)
    importlib.import_module(TOOLS[sys.argv[1]]).main(sys.argv[2:])
