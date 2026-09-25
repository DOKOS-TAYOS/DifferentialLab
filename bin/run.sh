#!/usr/bin/env bash
# ============================================================================
# DifferentialLab - Quick Launch Script for Unix/Mac
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Please run bin/setup.sh first"
    exit 1
fi

ENTRY_POINT=".venv/bin/differential-lab"
if [ ! -x "$ENTRY_POINT" ]; then
    echo "ERROR: DifferentialLab entry point not found or not executable: $ENTRY_POINT"
    echo "Please run bin/setup.sh first"
    exit 1
fi

MODE="${1:---prod}"

case "$MODE" in
    --dev|-d)
        "$ENTRY_POINT"
        ;;
    --background|-b|--prod|-p)
        mkdir -p logs
        nohup "$ENTRY_POINT" >> logs/run.log 2>&1 &
        PID="$!"
        echo "DifferentialLab started in background (PID: $PID)"
        echo "Logs: logs/run.log"
        ;;
    *)
        echo "Usage: bin/run.sh [--dev|-d|--background|-b|--prod|-p]"
        exit 1
        ;;
esac
