from __future__ import annotations

from attrs import field, frozen
from attrs.validators import ge, gt, in_, le, lt

DRIVERS: tuple[str, ...] = ("BAP1", "SF3B1", "EIF1AX", "GNAQ", "GNA11", "monosomy3", "chr8q")
COHORTS: tuple[str, ...] = ("TCGA-UVM", "GSE22138", "GSE44295", "GSE84976")

_ROUTER_BACKBONES: tuple[str, ...] = (
    "opus-4.7",
    "sonnet-4.6",
    "gemini-2.5-pro",
    "gpt-4o",
    "llama-4-maverick",
    "llama-3.1-70b",
    "qwen-2.5-72b",
)
_VLM_BACKBONES: tuple[str, ...] = (
    "conch",
    "virchow2",
    "prov-gigapath",
    "virchow",
    "uni",
    "gigapath",
    "pathchat-v2",
)


@frozen(kw_only=True)
class GraphConfig:
    hop_budget: int = field(default=3, validator=[ge(1), le(6)])
    string_tau: float = field(default=0.7, validator=[ge(0.0), le(1.0)])
    class2_expand_hops: int = field(default=2, validator=ge(0))


@frozen(kw_only=True)
class CohortConfig:
    sizes: dict[str, int] = field(
        factory=lambda: {"TCGA-UVM": 80, "GSE22138": 63, "GSE44295": 57, "GSE84976": 28}
    )
    bap1_null_rate: float = field(default=0.491, validator=[ge(0.0), le(1.0)])
    monosomy3_rate: float = field(default=0.518, validator=[ge(0.0), le(1.0)])
    class2_rate: float = field(default=0.539, validator=[ge(0.0), le(1.0)])
    gnaq_rate: float = field(default=0.588, validator=[ge(0.0), le(1.0)])
    gna11_rate: float = field(default=0.360, validator=[ge(0.0), le(1.0)])
    sf3b1_rate: float = field(default=0.20, validator=[ge(0.0), le(1.0)])
    eif1ax_rate: float = field(default=0.18, validator=[ge(0.0), le(1.0)])
    wsi_coverage: float = field(default=0.70, validator=[ge(0.0), le(1.0)])
    censoring_rate: float = field(default=0.55, validator=[ge(0.0), le(1.0)])
    risk_scale: float = field(default=2.4, validator=gt(0.0))
    class_signal: float = field(default=5.0, validator=ge(0.0))
    n_genes: int = field(default=15, validator=ge(1))
    seed: int = field(default=68)


@frozen(kw_only=True)
class ModelConfig:
    rna_dim: int = field(default=15, validator=ge(1))
    subtype_hidden: int = field(default=64, validator=ge(1))
    tile_dim: int = field(default=32, validator=ge(1))
    vlm_regions: int = field(default=16, validator=ge(1))
    vlm_embed: int = field(default=48, validator=ge(1))
    router_weights: dict[str, float] = field(
        factory=lambda: {
            "subtype": 1.0,
            "pathway": 0.8,
            "pathology": 0.9,
            "therapy": 0.5,
        }
    )


@frozen(kw_only=True)
class ConformalConfig:
    alpha: float = field(default=0.10, validator=[gt(0.0), le(0.5)])
    cal_fraction: float = field(default=0.386, validator=[gt(0.0), lt(1.0)])
    width_budget: float = field(default=6.6, validator=gt(0.0))
    pe_threshold: int = field(default=4, validator=ge(0))
    quantile_steps: int = field(default=300, validator=ge(1))
    quantile_lr: float = field(default=0.05, validator=gt(0.0))


@frozen(kw_only=True)
class TrainConfig:
    subtype_steps: int = field(default=400, validator=ge(1))
    subtype_lr: float = field(default=0.01, validator=gt(0.0))
    weight_decay: float = field(default=1e-4, validator=ge(0.0))
    seed: int = field(default=42)


@frozen(kw_only=True)
class EvalConfig:
    bootstrap: int = field(default=1000, validator=ge(1))
    mcid: float = field(default=0.05, validator=ge(0.0))
    seeds: tuple[int, ...] = field(
        default=(42, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    )
    ece_bins: int = field(default=10, validator=ge(1))
    holm_alpha: float = field(default=0.05, validator=[gt(0.0), lt(1.0)])
    bh_q: float = field(default=0.10, validator=[gt(0.0), lt(1.0)])
    net_benefit_thresholds: tuple[float, ...] = field(default=(0.10, 0.25, 0.40))


@frozen(kw_only=True)
class ExperimentConfig:
    name: str = field(default="main")
    agents: tuple[str, ...] = field(
        default=("subtype", "pathway", "pathology", "therapy", "router", "conformal")
    )
    kg_first: bool = field(default=True)
    router_backbone: str = field(default="opus-4.7", validator=in_(_ROUTER_BACKBONES))
    vlm_backbone: str = field(default="conch", validator=in_(_VLM_BACKBONES))
    holdout_cohort: str | None = field(default=None)
    graph: GraphConfig = field(factory=GraphConfig)
    cohort: CohortConfig = field(factory=CohortConfig)
    model: ModelConfig = field(factory=ModelConfig)
    conformal: ConformalConfig = field(factory=ConformalConfig)
    train: TrainConfig = field(factory=TrainConfig)
    eval: EvalConfig = field(factory=EvalConfig)
