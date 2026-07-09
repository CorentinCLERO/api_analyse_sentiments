#!/usr/bin/env bash
#
# Retrain the sentiment model from the latest data in the `tweets` table,
# then log the outcome.
#
# Manual run:   ./scripts/retrain.sh
# Cron run:     see README.md ("Automated retraining" section)
#
set -euo pipefail

# Project root = parent of this script's directory. Resolving it this way keeps
# the script working regardless of the CWD, which matters when cron runs it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/retrain.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== Retraining started ==="

# Pick the runner: uv if available (the project's tool), else a venv, else python3.
if command -v uv >/dev/null 2>&1; then
    RUNNER=(uv run python train_model.py)
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    RUNNER=("$PROJECT_DIR/.venv/bin/python" train_model.py)
else
    RUNNER=(python3 train_model.py)
fi

log "Command: ${RUNNER[*]}"

if "${RUNNER[@]}" >>"$LOG_FILE" 2>&1; then
    log "=== Retraining finished successfully ==="
    exit 0
else
    STATUS=$?
    log "=== Retraining FAILED (exit $STATUS) — see above ==="
    exit "$STATUS"
fi
