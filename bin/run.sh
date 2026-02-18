#!/bin/bash
# ============================================================================
# ODE Solver - Quick Launch Script for Unix/Mac
# ============================================================================

set -e

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Please run bin/setup.sh first"
    exit 1
fi

source .venv/bin/activate && nohup python src/main_program.py > /dev/null 2>&1 &
exit 0
