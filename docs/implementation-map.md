# Implementation Map

Every numbered equation, algorithm, theorem, reported table, ablation and metric mapped to the
module that carries it. Columns: paper anchor | item | file | notes.

## Definitions, equations, algorithms

| Anchor | Item | File | Notes |
|---|---|---|---|
| Def. 1 | Pathway knowledge graph as labelled directed multigraph G=(V,E,ℓ_V,ℓ_E) with provenance set Π | `precedent/graph.py` | adjacency + typed-edge store; provenance tags Reactome/OmniPath/STRING/KEGG |
| Def. 2 | KG-first multi-agent system tuple; partial order A_mol≺A_kg≺{A_path,A_tx}≺A_rt≺A_cp | `bench/order.py` | order encoded as a topological gate, validated at runtime |
| Def. 3 | Conformal predictor with coverage; q̂ = ⌈(n+1)(1−α)⌉/n order statistic | `ruling/conformal.py` | split-conformal core |
| Eq. 1 | S_KG(m,c) = (V'(m,c), E'(m,c)) mutation-conditioned subgraph; D(m)=V'∩V_drug | `precedent/subgraph.py` | Algorithm 2 body |
| Eq. 2 | h_σ(x_wsi) = CONCH(x_wsi | σ(m,c)) pathway-conditioned evidence map | `witnesses/pathology.py` | offline VLM surrogate, contract-preserving (D1) |
| Eq. 3 | 1−α ≤ P(y∈C(x)) ≤ 1−α + 1/(n+1) marginal coverage | `ruling/coverage.py` | Theorem 4 bound checker |
| Eq. 4 | KG-first hallucination bound E[H(M,x)] ≤ ρ⁺ ≤ E[H(M',x)] ≤ η | `ruling/coverage.py` | Theorem 5; ρ⁺ = ρ + ε_VLM(1−ρ) |
| Eq. 5 | cohort-stratified marginal coverage, n_min strata bound | `ruling/conformal.py` | Theorem 6 stratified predictor |
| Alg. 1 | Multi-Agent Orchestration with KG-first invariant; shared store S; trace π(x); refer-flag φ(x) | `bench/orchestrator.py` | end-to-end run over a patient tuple |
| Alg. 2 | Mutation-conditioned subgraph extraction; seeds=mutated genes; BFS k≤3; STRING ≥τ; class-2 expansion; rank-by-path-then-IC50 | `precedent/subgraph.py` | deterministic given (V',E',D) |
| Alg. 3 | Split-conformal quantile regression for treatment prediction sets; nonconformity; φ(x) fires on width/coverage | `ruling/conformal.py` | Romano CQR / Angelopoulos–Bates adaptive |

## Witness agents (§4.3)

| Anchor | Item | File | Notes |
|---|---|---|---|
| §4.3 | Molecular-Subtype agent A_mol — 15-gene DecisionDx-UM head → Class 1/2 + driver vector m | `witnesses/subtype.py` | torch MLP head |
| §4.3 | Pathway-KG agent A_kg — subgraph construction + bi-directional refinement | `witnesses/pathway.py` | wraps `precedent/` |
| §4.3 / Eq. 2 | Pathology-VLM agent A_path — region evidence map, abstains when x_wsi absent | `witnesses/pathology.py` | surrogate (D1) |
| §4.3 | Therapy-Evidence agent A_tx — OncoKB tier + DrugBank + GDSC2/CTRP IC50 + ClinicalTrials, ranked with provenance | `witnesses/therapy.py` | evidence aggregation over D(m) |
| §4.3 | Router A_rt — preference aggregation, shared evidence store S | `bench/router.py` | weighted aggregation (Table 10 default) |
| §4.3 | Conformal Calibration Layer A_cp | `ruling/conformal.py` | |

## Baselines & statistics

| Anchor | Item | File | Notes |
|---|---|---|---|
| §4.5 / Table 1 | DecisionDx-UM GEP, AJCC TNM 8th, BAP1 IHC, monosomy-3 FISH/SNP, PORPOISE, Pathomic-Fusion, MOGONet | `findings/baselines.py` | independent reimplementation heads |
| §4.8 | Harrell C-index, Uno C, 1000-bootstrap stratified CI | `findings/concordance.py` | survival concordance |
| §4.8 | paired DeLong test for correlated C-indices | `findings/delong.py` | |
| §4.8 | DerSimonian–Laird random-effects pooling + Cochran Q / I² | `findings/pooling.py` | inverse-variance weights |
| §4.8 / Table 6 | decision-curve net benefit (Vickers–Elkin), Brier, ECE 10 equal-mass bins, Kaplan–Meier log-rank | `findings/clinical.py` | |
| Table 2 | interpretability Likert + Krippendorff α | `findings/agreement.py` | inter-rater agreement |
| §4.8 | Holm–Bonferroni, Benjamini–Hochberg FDR | `findings/multiplicity.py` | family-wise / FDR control |

## Reported tables / experiments → configs

Runnable configs drive `run_main`; tables that are slices of the same run are produced by config
flags (`agents`, `kg_first`, `holdout_cohort`, `router_backbone`, `vlm_backbone`) or read off the main
report rather than a separate file.

| Table | Experiment | How produced |
|---|---|---|
| Table 1 | four-cohort metastasis-risk stratification | `experiment/main.yaml` |
| Table 2 | agent-architecture ablation | `ablation_architecture.yaml` (kg_first off), `ablation_drop_pathology.yaml`, `ablation_drop_pathway.yaml`, or `--set agents=[...]` |
| Table 3 | subgroup performance | stratify `main` report by `drivers`/`gep_class` |
| Table 4 | leave-one-cohort-out robustness | `ablation_loco.yaml` (+ `--set holdout_cohort=...`) |
| Table 5 | per-agent cost / tool-call | recorded in `configs/experiment/main.yaml` provenance (Table 5) |
| Table 6 | net benefit / calibration / survival | `main.yaml` eval block (`MainReport.net_benefit/brier/ece`) |
| Table 7 | per-seed variance (10 seeds) | `evaluate --seed` over `eval.seeds` |
| Table 8 | cross-site concordance matrix | `ablation_loco.yaml` across the four holdouts |
| Table 9 | scaling (agent count / backbone) | `--set agents=[...]` / `supplementary_router_backbone.yaml` |
| Table 10 | ensemble aggregation comparison | `bench/router.py` weighted aggregation (default) |
| Table 11 | Router LLM backbone ablation | `supplementary_router_backbone.yaml` (+ `--set router_backbone=...`) |
| Table 12 | pathology-VLM backbone ablation | `supplementary_vlm_backbone.yaml` (+ `--set vlm_backbone=...`) |
| Table 13 | adversarial prompt-injection robustness | `bench/order.py` invariant + conformal refer-flag (qualitative) |

## Findings F1–F4

| Finding | Where validated |
|---|---|
| F1 KG-first dissociates discrimination from interpretability | `experiment/ablation_architecture.yaml` rows 2,5 + `findings/agreement.py` |
| F2 BAP1-null mechanistic concordance (43/50) | `witnesses/pathway.py` + `experiment/ablation_subgroup.yaml` |
| F3 cross-site narrow-band generalisation (≤2.6pp) | `findings/pooling.py` + `experiment/ablation_crosssite.yaml` |
| F4 conformal refer-for-expert-review as actionable signal (7.2%) | `ruling/conformal.py` refer-flag decomposition |
