#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p workspace
"$PYTHON_BIN" agent.py > workspace/results.log 2>&1
