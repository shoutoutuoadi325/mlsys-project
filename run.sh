#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
	if [[ -d /usr/local/cuda/bin ]]; then
		export PATH="/usr/local/cuda/bin:${PATH}"
	elif [[ -d /usr/local/cuda-12.4/bin ]]; then
		export PATH="/usr/local/cuda-12.4/bin:${PATH}"
	fi
fi

python3 -m agent.main
