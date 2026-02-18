#!/bin/bash
# ============================================================================
# ODE Solver - Clean temporary files (Unix/Mac)
# Removes __pycache__, .pyc, .pyo, .log, and other temp files.
# Skips the .venv directory.
# ============================================================================

set -e

cd "$(dirname "$0")/.."

echo "Cleaning temporary files..."

count=0

# Remove __pycache__ directories
while IFS= read -r -d '' dir; do
    rm -rf "$dir"
    ((count++))
done < <(find . -path ./.venv -prune -o -type d -name "__pycache__" -print0 2>/dev/null | grep -z -v "^./\.venv")

# Remove .pyc and .pyo files
while IFS= read -r -d '' file; do
    rm -f "$file"
    ((count++))
done < <(find . -path ./.venv -prune -o -type f \( -name "*.pyc" -o -name "*.pyo" \) -print0 2>/dev/null | grep -z -v "^./\.venv")

# Remove .mypy_cache, .pytest_cache, .ruff_cache
for cache_dir in .mypy_cache .pytest_cache .ruff_cache; do
    while IFS= read -r -d '' dir; do
        rm -rf "$dir"
        ((count++))
    done < <(find . -path ./.venv -prune -o -type d -name "$cache_dir" -print0 2>/dev/null | grep -z -v "^./\.venv")
done

# Remove egg-info directories
while IFS= read -r -d '' dir; do
    rm -rf "$dir"
    ((count++))
done < <(find . -path ./.venv -prune -o -type d -name "*.egg-info" -print0 2>/dev/null | grep -z -v "^./\.venv")

# Remove log files from root
for f in ./*.log; do
    [ -e "$f" ] && rm -f "$f" && ((count++))
done

echo "Done. Cleaned $count items."
