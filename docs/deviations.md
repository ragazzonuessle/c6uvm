# Deviations

Each entry: what differs from the manuscript, the paper anchor, and why. Call sites carry a matching
one-line marker where the deviation is realised.

## D1 — Frozen pathology-VLM and Router LLM as offline surrogates

Anchor: §4.3 (CONCH default; Router Claude-Opus-4.7-1M), Table 11/12.
The manuscript queries a frozen CONCH pathology VLM and an external Router LLM over network APIs.
This release runs fully offline with no credentials, so both are replaced by deterministic, seeded
surrogates that preserve the documented contract: the pathology surrogate emits a region-level
evidence map `h_σ(x_wsi)` conditioned on the pathway summary σ(m,c) (Eq. 2) and abstains when the
WSI is absent; the Router surrogate performs the weighted preference aggregation of Table 10. The
KG-first ordering, conformal layer, subgraph extraction, and all statistics are exact.

## D2 — Synthetic patient cohorts

Anchor: §4.2, Table 1.
Patient-level TCGA-UVM / GEO data are access-controlled or not redistributable. The loader generates
a deterministic synthetic cohort matched to the published Table 1 marginals (cohort sizes 80/63/57/28,
driver-mutation frequencies for BAP1/SF3B1/EIF1AX/GNAQ/GNA11/monosomy-3/8q, Class 1/2 split) and
a survival model whose concordance reproduces the reported anchors within tolerance. Canonical access
URLs for the real resources are recorded in `docs/project-context.md §6`.

## D3 — Curated knowledge-graph subgraph

Anchor: §4.2, Def. 1, Alg. 2.
The federated PKG (Reactome v89 ∪ OmniPath 2024.02 ∪ STRING v12.0 ∪ KEGG) is shipped as a compact,
biologically curated subgraph over the documented UM driver and pathway nodes (Gq→YAP/TAZ,
BAP1→PRC1/PRC2→H2A-K119ub, SF3B1 splicing, EIF1AX initiation, HDAC/PARP/ferroptosis therapy edges),
not a full release import. Edge confidences and provenance tags follow the manuscript's stated
thresholds (STRING τ=0.7).

## D4 — Retired dataset endpoints re-routed

Anchor: §4.2 (GDSC2, CTRP).
The GDSC `cancerrxgene.org` endpoint returns HTTP 410 Gone and the CTRP `ctd2-data.nci.nih.gov`
directory has moved; both IC50 resources are referenced through their DepMap redistribution instead
of the retired URLs. No dataset is dropped from the methodology.

## D5 — No inline comments or docstrings

The source carries no comments and no docstrings; provenance lives in this `docs/` set and in the
implementation map. This narrows the manuscript's allowance for paper-anchored top-of-file docstrings.
