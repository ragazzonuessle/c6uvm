# Repository Plan

## Layout

```
c6_uvm/
├── pyproject.toml          # build + tool config (ruff/black/isort/mypy/pytest)
├── requirements.txt
├── environment.yml
├── Dockerfile
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE                 # MIT
├── configs/
│   ├── model/c6.yaml
│   ├── data/cohorts.yaml
│   ├── train/calibration.yaml
│   └── experiment/{main,_smoke,ablation_*,supplementary_*}.yaml
├── src/c6_uvm/
│   ├── intake/             # synthetic cohort generator + typed patient tuples
│   ├── record/             # shared evidence store S, provenance trace
│   ├── precedent/          # pathway KG, mutation-conditioned BFS subgraph (Alg.2)
│   ├── witnesses/          # the four substantive agents
│   ├── bench/              # Router + KG-first order gate + orchestrator (Alg.1)
│   ├── ruling/             # split-conformal layer + coverage/hallucination bounds
│   ├── findings/           # baselines + concordance + statistical tests
│   ├── proceedings/        # calibration / evaluation drivers, seeding, checkpoints
│   ├── clerk/              # typer CLI: try / eval / infer / export
│   └── statute/            # attrs config schema + cattrs YAML structuring
├── scripts/{launch_train,launch_eval,prepare_data}.sh
├── tests/
└── docs/
```

## Module responsibilities

- `statute` — frozen `attrs` config dataclasses; `cattrs` converter structures layered YAML into them;
  CLI dotlist overrides applied before structuring.
- `intake` — `CohortSpec` → deterministic synthetic patients (RNA vector, driver indicators, optional
  WSI tile tensor, survival time/event) matched to Table 1 marginals.
- `record` — `EvidenceStore` keyed by subgraph node id; append-only; reconstructs trace π(x).
- `precedent` — `PathwayGraph` adjacency over UM driver/pathway nodes; `extract_subgraph` is Alg. 2.
- `witnesses` — `SubtypeAgent`, `PathwayAgent`, `PathologyAgent`, `TherapyAgent`; each consumes the
  patient tuple / shared store and writes typed evidence.
- `bench` — `kg_first_order` gate (Def. 2), `Router` aggregation, `Orchestrator.run` (Alg. 1).
- `ruling` — `SplitConformal` (Alg. 3), `marginal_band` / `hallucination_bound` (Thm. 4/5),
  `StratifiedConformal` (Thm. 6).
- `findings` — concordance, DeLong, DerSimonian–Laird, net benefit, Brier, ECE, Krippendorff,
  multiplicity control, baseline heads.
- `proceedings` — `calibrate`, `evaluate`, seed utility, atomic checkpoint I/O.
- `clerk` — CLI verbs binding config → proceedings.

## Pinned dependencies

| package | pin | role |
|---|---|---|
| python | 3.11–3.14 | runtime |
| torch | 2.12.* | VLM/subtype/baseline heads |
| numpy | ≥1.26 | arrays |
| scipy | ≥1.11 | special functions, stats |
| lifelines | ≥0.29 | survival concordance / KM |
| attrs | ≥24 | config dataclasses |
| cattrs | ≥24 | YAML → struct |
| typer | ≥0.12 | CLI |
| pyyaml | ≥6 | config IO |

Dev: pytest, hypothesis, ruff, black, isort, mypy, pre-commit.

## Test coverage targets

- `precedent`: BFS determinism, hop-budget invariant, class-2 expansion, seeds-are-mutated-only.
- `ruling`: empirical coverage ∈ band on synthetic exchangeable data (property-based, Hypothesis);
  monotone width vs α; refer-flag fires on width/coverage breach; stratified ≥ marginal.
- `witnesses`: subtype head overfits a single batch; pathology agent abstains without WSI.
- `findings`: DeLong matches a closed-form 2-sample case; DSL pooling reduces to fixed-effect at I²=0;
  ECE/Brier bounds; Krippendorff α on a known table.
- `bench`: order gate rejects an out-of-order schedule; orchestrator trace reconstructs evidence.
- e2e: `tests/test_training_smoke.py` runs 2 calibration steps on `_smoke.yaml`, loss decreases.
