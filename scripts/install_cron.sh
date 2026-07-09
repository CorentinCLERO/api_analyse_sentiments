#!/usr/bin/env bash
#
# Install (or refresh) the weekly retraining cron entry for the current user.
# Idempotent: re-running it replaces the existing entry instead of duplicating.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRAIN="$SCRIPT_DIR/retrain.sh"

# Every Monday at 03:00.
SCHEDULE="0 3 * * 1"
MARKER="# api_analyse_sentiments weekly retrain"
ENTRY="$SCHEDULE $RETRAIN $MARKER"

# Keep every existing line except a previous entry of ours, then append the fresh one.
( crontab -l 2>/dev/null | grep -vF "$MARKER" || true; echo "$ENTRY" ) | crontab -

echo "Cron entry installed:"
echo "  $ENTRY"
echo
echo "Current crontab:"
crontab -l
