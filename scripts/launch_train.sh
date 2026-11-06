#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-configs/experiment/main.yaml}"
OUT="${2:-checkpoints/$(basename "${EXPERIMENT%.yaml}").pt}"

python -m c6_uvm.clerk.app train --experiment "${EXPERIMENT}" --out "${OUT}"
