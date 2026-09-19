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

MODE="${1:---prod}"

case "$MODE" in
    --dev|-d)
        source .venv/bin/activate
        python src/main_program.py
        ;;
    --background|-b|--prod|-p)
        source .venv/bin/activate
        mkdir -p logs
        nohup python src/main_program.py >> logs/run.log 2>&1 &
        PID="$!"
        echo "DifferentialLab started in background (PID: $PID)"
        echo "Logs: logs/run.log"
        ;;
    *)
        echo "Usage: bin/run.sh [--dev|-d|--background|-b|--prod|-p]"
        exit 1
        ;;
esac
