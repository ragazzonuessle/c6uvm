#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-configs/experiment/main.yaml}"

python -m c6_uvm.clerk.app evaluate --experiment "${EXPERIMENT}"
