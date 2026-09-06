#!/bin/bash
#
# Submit the eddy-tracking pipeline as chained Slurm jobs.
#
# Usage:
#   ./slurm/submit_pipeline.sh <experiment> # all stages
#   ./slurm/submit_pipeline.sh <experiment> --from <stage> # from stage onward
#
# Each stage submits with --dependency=afterok:<prev_job_id> so it only
# starts after the previous stage succeeds. If any stage fails, all
# downstream stages stay in "pending (dependency)" and can be cancelled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
EXPERIMENT=""
FROM_STAGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            FROM_STAGE="$2"
            shift 2
            ;;
        *)
            EXPERIMENT="$1"
            shift
            ;;
    esac
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "Usage: $0 <experiment> [--from <stage>]"
    echo "Stages: download_swot download_pace download_sst_sss eddy_id eddy_track collocate_pace run_sdp"
    exit 1
fi

# Verify config exists
if [[ ! -d "$PROJECT_DIR/configs/$EXPERIMENT" ]]; then
    echo "ERROR: Config directory not found: configs/$EXPERIMENT"
    exit 1
fi

# Create logs directory on the cluster
mkdir -p "$PROJECT_DIR/logs"

# Ordered stages - downloads run sequentially first, then compute
ALL_STAGES=(download_swot download_pace download_sst_sss eddy_id eddy_track collocate_pace run_sdp)

# Determine which stages to run
STAGES=()
FOUND=false
for stage in "${ALL_STAGES[@]}"; do
    if [[ -n "$FROM_STAGE" ]]; then
        if [[ "$stage" == "$FROM_STAGE" ]]; then
            FOUND=true
        fi
        if $FOUND; then
            STAGES+=("$stage")
        fi
    else
        STAGES+=("$stage")
    fi
done

if [[ -n "$FROM_STAGE" && "$FOUND" == false ]]; then
    echo "ERROR: Unknown stage '$FROM_STAGE'"
    echo "Valid stages: ${ALL_STAGES[*]}"
    exit 1
fi

echo "========================================"
echo "  Pipeline: $EXPERIMENT"
echo "  Stages:   ${STAGES[*]}"
echo "========================================"

# Download stages run in parallel; all others run sequentially after them.
DOWNLOAD_STAGES=(download_swot download_pace download_sst_sss)
DOWNLOAD_JIDS=()

PREV_JID=""
for stage in "${STAGES[@]}"; do
    SBATCH_FILE="$SCRIPT_DIR/${stage}.sbatch"
    if [[ ! -f "$SBATCH_FILE" ]]; then
        echo "ERROR: Missing sbatch file: $SBATCH_FILE"
        exit 1
    fi

    CMD="sbatch --parsable --export=ALL,EXPERIMENT=$EXPERIMENT"

    # Downloads run in parallel (no dependency on each other).
    # The first sequential stage (eddy_id) depends on all three completing.
    is_download=false
    for dl in "${DOWNLOAD_STAGES[@]}"; do
        [[ "$stage" == "$dl" ]] && is_download=true && break
    done

    if $is_download; then
        : # no dependency - submit immediately
    elif [[ ${#DOWNLOAD_JIDS[@]} -gt 0 && -z "$PREV_JID" ]]; then
        # First sequential stage: depend on all completed downloads
        DEP=$(IFS=:; echo "${DOWNLOAD_JIDS[*]}")
        CMD="$CMD --dependency=afterok:$DEP"
    elif [[ -n "$PREV_JID" ]]; then
        CMD="$CMD --dependency=afterok:$PREV_JID"
    fi

    CMD="$CMD $SBATCH_FILE"
    JID=$($CMD)

    if $is_download; then
        DOWNLOAD_JIDS+=("$JID")
        echo "  $stage -> job $JID (parallel download)"
    else
        echo "  $stage -> job $JID${PREV_JID:+ (after $PREV_JID)}"
        PREV_JID="$JID"
    fi
done

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/<stage>_<jobid>.log"
