#!/usr/bin/env bash
set -euo pipefail

mkdir -p workspace
python3 agent.py > workspace/results.log 2>&1
