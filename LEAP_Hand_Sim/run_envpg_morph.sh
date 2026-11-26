#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_SCRIPT="${SCRIPT_DIR}/sb3_train_envpg_morph.py"
CHECKPOINT="${SCRIPT_DIR}/leapsim/envpg_logs/checkpoints/envpg_physical_resume.pt"

ARGS=()

SKIP_NEXT=0

for ARG in "$@"; do
    if [[ "$SKIP_NEXT" -eq 1 ]]; then
        SKIP_NEXT=0
        continue
    fi

    if [[ "$ARG" == "--resume-checkpoint" ]]; then
        SKIP_NEXT=1
        continue
    fi

    if [[ "$ARG" == --resume-checkpoint=* ]]; then
        continue
    fi

    ARGS+=("$ARG")
done

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

while true; do

    if [[ -f "$CHECKPOINT" ]]; then

        echo "[LAUNCHER] Resuming from checkpoint:"
        echo "$CHECKPOINT"

        python "$TRAIN_SCRIPT" \
            "${ARGS[@]}" \
            --resume-checkpoint "$CHECKPOINT"

    else

        echo "[LAUNCHER] Starting fresh training run."

        python "$TRAIN_SCRIPT" "${ARGS[@]}"

    fi

    RC=$?

    if [[ "$RC" -eq 42 ]]; then

        echo "[LAUNCHER] Isaac Gym process exited for morphology restart."
        echo "[LAUNCHER] Starting a clean resume process..."

        sleep 2

        continue

    fi

    echo "[LAUNCHER] Training process exited with code $RC"

    exit "$RC"

done