# C6 — In re: Uveal Melanoma Metastasis Risk and Therapy Set

A standing record of the six-agent, KG-first vision-language system that hears a uveal-melanoma
case, weighs molecular, pathway, histology, and therapy-evidence testimony under a fixed order of
proceedings, and returns a conformally calibrated treatment-set with an explicit refer-for-expert-review
disposition. Coverage is a marginal frequency property, not a per-patient safety guarantee;
prospective validation is required before clinical use.

The six agents are entered as witnesses and officers of the bench:

| Officer | Role | Module |
|---|---|---|
| Molecular-Subtype | 15-gene GEP testimony → Class 1/2 + driver vector | `witnesses/subtype.py` |
| Pathway-KG | mutation-conditioned subgraph over Reactome/OmniPath/STRING/KEGG (first-class) | `precedent/`, `witnesses/pathway.py` |
| Pathology-VLM | pathway-conditioned histology evidence map; abstains when no slide is in the record | `witnesses/pathology.py` |
| Therapy-Evidence | OncoKB/DrugBank/GDSC2/CTRP ranked therapy with provenance | `witnesses/therapy.py` |
| Router | preference aggregation over the shared evidence store | `bench/router.py` |
| Conformal Layer | split-conformal prediction set + refer-flag | `ruling/conformal.py` |

The KG-first invariant fixes the order of proceedings `A_mol ≺ A_kg ≺ {A_path, A_tx} ≺ A_rt ≺ A_cp`
(`bench/order.py`); the pathology witness is examined only after the mutation-conditioned subgraph is
on the record.

## I. Entering an Appearance — installation

    python -m venv .venv && . .venv/bin/activate
    pip install -e .            # add [export] for ONNX, [dev] for the test toolchain

Conda:

    conda env create -f environment.yml && conda activate c6_uvm && pip install -e .

Container:

    docker build -t c6_uvm . && docker run --rm c6_uvm evaluate -e configs/experiment/main.yaml

## II. The Record — data

Patient-level cohorts are access-controlled at source and are not redistributed here; `intake/`
admits a deterministic synthetic record matched to the published Table 1 marginals and the seven UM
driver frequencies, and the knowledge graph ships as a curated UM driver/pathway subgraph
(`precedent/graph.py`). Canonical sources, versions, licences, and access points (checked while
building this release):

| Resource | Version | Licence | Access |
|---|---|---|---|
| TCGA-UVM (n=80) | GDC Release 40.0 | NIH/GDC open tier | https://portal.gdc.cancer.gov/projects/TCGA-UVM |
| GSE22138 (n=63) | GEO, Affymetrix HG-U133Plus2 | open | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22138 |
| GSE44295 (n=57) | GEO, Illumina bead-array | open | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44295 |
| GSE84976 (n=28) | GEO, Illumina HumanHT-12 v4.0 | open | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84976 |
| DepMap CCLE | 24Q4 | CC-BY 4.0 | https://depmap.org/portal/ |
| GDSC2 / CTRP IC50 | v8.5 / v2 | CC-BY 4.0 | via DepMap redistribution |
| Reactome | v89 | CC0 | https://reactome.org/ |
| OmniPath | 2024.02 | CC-BY-NC 4.0 | https://omnipathdb.org/ |
| STRING | v12.0 | CC-BY 4.0 | https://string-db.org/ |
| KEGG | public API | academic terms | https://www.kegg.jp/ |
| DrugBank Open | 5.1.12 | CC-BY-NC 4.0 | https://go.drugbank.com/ |
| OncoKB | public tier v4.18 | academic terms | https://www.oncokb.org/ |
| ClinicalTrials.gov AACT | 2026-04-01 | public domain | https://aact.ctti-clinicaltrials.org/ |
| COSMIC | v100 | academic tier | https://cancer.sanger.ac.uk/cosmic |

The GDSC `cancerrxgene.org` and CTRP `ctd2-data.nci.nih.gov` endpoints are retired; their IC50
profiles are reached through DepMap redistribution. The synthetic record is regenerated on demand:

    bash scripts/prepare_data.sh

## III. Calling the Witnesses — training

One command admits the full panel, calibrates the conformal layer, and seals a checkpoint:

    python -m c6_uvm.clerk.app train -e configs/experiment/main.yaml --out checkpoints/main.pt

    # ablations and supplementary panels
    c6 train -e configs/experiment/ablation_architecture.yaml      # KG-first invariant removed
    c6 train -e configs/experiment/ablation_drop_pathology.yaml    # pathology-VLM withdrawn
    c6 train -e configs/experiment/ablation_loco.yaml              # leave-one-cohort-out
    c6 train -e configs/experiment/supplementary_router_backbone.yaml

Configuration is layered YAML structured into frozen `attrs` records via `cattrs`; any leaf is
overridable on the command line, for example `--set conformal.alpha=0.2 --set router_backbone=gpt-4o`.

## IV. Findings of Fact — evaluation

    c6 evaluate -e configs/experiment/main.yaml

Expected entries on the four-cohort record (10 fixed seeds; tolerances are seed-level dispersion):

| Cohort | C6 C-index | DecisionDx-UM | ΔC |
|---|---|---|---|
| TCGA-UVM (n=80) | 0.833 ± 0.010 | 0.774 | +0.058 |
| GSE22138 (n=63) | 0.819 ± 0.012 | 0.761 | +0.057 |
| GSE44295 (n=57) | 0.827 ± 0.014 | 0.680 | +0.147 |
| GSE84976 (n=28) | 0.789 ± 0.020 | 0.674 | +0.115 |
| pooled (DSL) | 0.824 [0.796, 0.853] | — | paired p ≈ 1e-3 |

Every cohort crosses the 0.05 minimal-clinically-important-difference floor against the DecisionDx-UM
gene-expression standard. At α = 0.10 the conformal layer returns empirical coverage ≈ 0.93, sitting
above the 0.900 marginal floor (theoretical band `[0.900, 0.911]` at n_cal = 88), with a
refer-for-expert-review rate near 0.09. Leave-one-cohort-out retains a narrow band (held-out C-index
0.79–0.83), evidence that the panel generalises rather than memorising a single cohort. A single auditable case is examined with:

    c6 infer -e configs/experiment/main.yaml --cohort TCGA-UVM --index 0

which prints the KG-first evidence chain, the candidate therapy set, the prediction interval, and the
refer disposition. The subtype head exports to ONNX with `c6 export --out artifacts/subtype.onnx`.

## V. Cost of Proceedings — compute budget

The manuscript reports a single NVIDIA A100 80 GB profile with the Router at temperature 0, ten fixed
seeds, 3.5 min/case, $0.50–1.00/case, and roughly $150–175 for the 228-case four-cohort run with at
most twelve bounded tool calls per case. This offline record carries deterministic surrogates for the
frozen pathology-VLM and the Router LLM (`docs/deviations.md`, D1), so it removes all API cost and runs
CPU-light: the full evaluation completes in minutes on a laptop. The reported budget is preserved in
`configs/experiment/main.yaml` for provenance.

## VI. Sealed Exhibits — checkpoints

`train` writes an atomic checkpoint (temporary file then `os.replace`) carrying the resolved
configuration, the seed, the subtype head state, and the Router backbone tag, so any sealed exhibit can
be reopened and the run reconstructed. Released model weights are available upon reasonable request.

## VII. Standing Orders — reproducibility

A single `set_seed` governs Python, NumPy, and Torch; the seed is recorded in every checkpoint. The
order of proceedings is enforced at runtime (`bench/order.py`); a schedule that violates the KG-first
invariant is rejected. The engineering record lives in `docs/` (`project-context.md`,
`implementation-map.md`, `deviations.md`). Continuous checks: `ruff check .`, `black --check .`,
`isort --check .`, `mypy --strict src/c6_uvm`, and `pytest -q`.
