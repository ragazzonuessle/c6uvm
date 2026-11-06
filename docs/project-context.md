# Project Context

Canonical engineering record for the C6 release accompanying the manuscript
*From Molecular Networks to Clinical Decisions: Multi-Agent Vision-Language Systems for
Systems Biology-Enabled Uveal Melanoma Therapy* (npj Precision Oncology).

    project_name       : c6_uvm                                     [HIGH]
    domain             : computational oncology — rare-cancer       [HIGH]
                         multi-agent vision-language decision support
                         (uveal melanoma metastasis-risk + therapy set)
    framework          : PyTorch 2.x + hand-rolled pathway-KG/BFS   [MED]
                         + split-conformal + survival analysis;
                         frozen-VLM / Router-LLM as offline surrogates
    venue              : npj Precision Oncology                     [HIGH]
    primary_datasets   : 15 public resources (see §6)               [HIGH]
    compute_target     : 1x A100 80GB reported; offline run is      [HIGH]
                         CPU-light (orchestration + small heads)
    hparams_reference  : §4.6 / §4.8 / §4.9 + Alg.2 + Alg.3         [HIGH]
    supp_path          : none (SI tables inlined as Tables 7-13)    [HIGH]

    NEEDS_USER_DECISION: 0
    Build mode         : local build only (no remote publication)

## 1. project_name — `c6_uvm` [HIGH]

The system is named **C6** throughout (Abstract; §4.3 "The C6 system orchestrates six agents").
Disease scope is uveal melanoma (UVM/UM). Package name binds the system identity to the cohort
domain. Confirmed by the user.

## 2. domain [HIGH]

Rare-cancer (uveal melanoma) metastasis-risk stratification and treatment-set decision support via a
six-agent vision-language pipeline coupling a pathology foundation model, a molecular-subtype
classifier, a federated pathway knowledge graph, and a therapy-evidence aggregator under a
conformal calibration layer (Abstract; §1 final paragraph; §4.3).

## 3. framework [MED]

PyTorch 2.x carries the pathology-VLM encoder, the 15-gene subtype head, and the baseline heads.
The pathway knowledge graph and its mutation-conditioned BFS extractor (Algorithm 2) are
hand-rolled over an adjacency representation. The conformal calibration layer (Algorithm 3) is pure
array math. Survival concordance uses standard estimators. The frozen pathology-VLM (CONCH) and
the Router LLM are network/credential-bound services in the manuscript; here they are replaced by
deterministic, seeded offline surrogates that preserve the documented input/output contract
(region-level evidence map for the VLM; per-agent preference aggregation for the Router). Logged as
deviations D1. Confidence MED because the surrogate boundary is an engineering choice, not a paper
value.

## 4. venue — npj Precision Oncology [HIGH]

Manuscript filename prefix and reference [75] (npj Precision Oncology editorial). Structured abstract,
four-cohort retrospective framing, TRIPOD+AI / DECIDE-AI checklists (§4.10) are consistent with the
npj clinical-AI format.

## 5. compute_target [HIGH]

§4.8 / Proposition 8 / Table 5: 1x NVIDIA A100 80GB, Router temperature 0, 10 fixed seeds, prompt
templates version-locked, 3.5 min/case, $0.50–1.00/case, ~$150–175 for the 228-case four-cohort run,
B ≤ 12 bounded API calls/case. The offline implementation removes all API cost and runs CPU-light;
the reported budget is recorded verbatim in `configs/experiment/main.yaml` for provenance.

## 6. primary_datasets [HIGH]

Access date 2026-04-20. Every link below was checked during this build; retired endpoints were
dropped and their data re-routed through a live redistributor (see §8 verification log).

Patient cohorts (N=228 across four institutions):
- TCGA-UVM — GDC Release 40.0, NIH/GDC open tier, n=80. https://portal.gdc.cancer.gov/projects/TCGA-UVM
- GSE22138 — GEO, Affymetrix HG-U133Plus2, Institut Curie, n=63. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22138
- GSE44295 — GEO, Illumina bead-array, Washington University, n=57. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44295
- GSE84976 — GEO, Illumina HumanHT-12 v4.0, Leiden UMC, n=28. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84976

Cell-line pharmacogenomics:
- DepMap CCLE 24Q4 (UM lines), CC-BY 4.0. https://depmap.org/portal/
- GDSC2 IC50 profiles, CC-BY 4.0 — accessed via DepMap redistribution (the cancerrxgene.org endpoint is retired; see §8).
- CTRP v2 sensitivity, CC-BY 4.0 — accessed via DepMap redistribution (the ctd2-data.nci.nih.gov endpoint moved; see §8).

Pathway / evidence knowledge bases:
- Reactome v89, CC0. https://reactome.org/
- OmniPath 2024.02, CC-BY-NC 4.0. https://omnipathdb.org/
- STRING v12.0, CC-BY 4.0. https://string-db.org/
- KEGG public API, academic terms. https://www.kegg.jp/
- DrugBank Open 5.1.12, CC-BY-NC 4.0. https://go.drugbank.com/
- OncoKB public tier v4.18, academic terms. https://www.oncokb.org/
- ClinicalTrials.gov AACT snapshot 2026-04-01, public domain. https://aact.ctti-clinicaltrials.org/
- COSMIC v100, academic tier. https://cancer.sanger.ac.uk/cosmic

Patient-level cohorts are not redistributed in this repository; the loader emits a synthetic cohort
matched to the published Table 1 marginal statistics and the seven UM driver frequencies (deviation
D2). The knowledge graph ships as a curated UM driver/pathway subgraph (deviation D3).

## 7. hparams_reference [HIGH]

No single hyperparameter table exists; the operative settings are distributed across:
- §4.6 — primary-metric MCID ΔC = 0.05; auxiliary MCIDs (HR ≥ 2.0, ΔConc@k ≥ 0.10, ΔAcc@5 ≥ 0.05, ΔTool ≥ 0.03).
- §4.8 — 1000-sample stratified bootstrap; paired DeLong; DerSimonian–Laird I²; ECE 10 equal-mass bins; Holm–Bonferroni α_family = 0.05; Benjamini–Hochberg q = 0.10; 10 fixed seeds; Router temperature 0.
- §4.9 / Theorem 6 — target miscoverage α = 0.10; calibration fold n = 88 of N = 228 (38.6%); cohort-stratified band [0.900, 0.921] at empirical 0.913.
- Algorithm 2 — BFS hop budget k = 3; STRING confidence τ = 0.7; class-2 BAP1-chromatin expansion.
- Algorithm 3 — width budget k_width; pathway-evidence coverage threshold τ_pe.
- Proposition 8 — message budget B ≤ 12, iteration cap T_max.

All of the above are encoded as defaults in `configs/experiment/main.yaml`.

## 8. extra_signals & verification log

- Algorithm boxes: 3 (Alg.1 orchestration, Alg.2 subgraph extraction, Alg.3 split-conformal).
- Theory: Definitions 1-3; Theorems 4 (marginal coverage), 5 (KG-first hallucination bound),
  6 (cohort-stratified coverage); Lemma 7 (reimplementation concordance).
- Pre-registered subgroup dimensions: 4 (BAP1 status, GNAQ/GNA11, monosomy-3/disomy-3, Class 1/2).
- Ablation matrix: 7 Router LLMs x 6 pathology FMs.
- Ethics firewall: §4.1 (retrospective, public de-identified data; advisory output; coverage ≠ safety).
- Link verification (checked this build):
  - live: TCGA-UVM/GDC, GSE22138, GSE44295, GSE84976, DepMap, STRING v12.0, OmniPath,
    KEGG, COSMIC, AACT.
  - live but bot/JS-guarded (kept): Reactome (HTTP 403), DrugBank (HTTP 403), OncoKB (client render).
  - retired/moved (dropped, re-routed via DepMap): GDSC cancerrxgene.org (HTTP 410 Gone),
    CTRP ctd2-data.nci.nih.gov (301 to a catalog stub).
