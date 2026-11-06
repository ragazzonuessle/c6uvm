#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-configs/experiment/main.yaml}"

python - "$EXPERIMENT" <<'PY'
import sys
from c6_uvm.intake.cohort import build_dataset, driver_frequencies
from c6_uvm.statute.config import DRIVERS
from c6_uvm.statute.loader import load_experiment

cfg = load_experiment(sys.argv[1])
data = build_dataset(cfg.cohort)
print("synthetic record admitted")
for name, cohort in data.items():
    freqs = driver_frequencies(cohort)
    summary = ", ".join(f"{d}={f:.2f}" for d, f in zip(DRIVERS, freqs, strict=True))
    print(f"  {name}: n={len(cohort)}  {summary}")
PY
