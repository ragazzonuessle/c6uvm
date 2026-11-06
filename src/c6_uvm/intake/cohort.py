from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from c6_uvm.record.types import Cohort, FloatArray, Patient
from c6_uvm.statute.config import CohortConfig

_BASIS_SEED = 20240617
_MORPH_SCALE = 1.35
_CENSOR_SCALE = 2.6
_ETA_CENTER = 1.05


def _unit(vec: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vec))
    scaled = vec / norm if norm > 0 else vec
    return np.asarray(scaled, dtype=np.float64)


def _basis(dim: int, key: int) -> FloatArray:
    rng = np.random.default_rng(_BASIS_SEED + key)
    return _unit(rng.normal(size=dim).astype(np.float64))


def morphology_basis(dim: int) -> FloatArray:
    return _basis(dim, 7)


EPITHELIOID_TEMPLATE: FloatArray = morphology_basis(32)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def generate_cohort(name: str, size: int, cfg: CohortConfig, seed: int) -> Cohort:
    rng = np.random.default_rng(seed)
    v_class = _basis(cfg.n_genes, 1)
    v_bap1 = _basis(cfg.n_genes, 2)
    tile_dim = EPITHELIOID_TEMPLATE.shape[0]
    patients: list[Patient] = []
    for idx in range(size):
        bap1 = 1.0 if rng.random() < cfg.bap1_null_rate else 0.0
        mono_p = float(np.clip(cfg.monosomy3_rate + 0.25 * (bap1 - 0.5), 0.0, 1.0))
        mono = 1.0 if rng.random() < mono_p else 0.0
        u = rng.random()
        gnaq = 1.0 if u < cfg.gnaq_rate else 0.0
        gna11 = 1.0 if cfg.gnaq_rate <= u < cfg.gnaq_rate + cfg.gna11_rate else 0.0
        sf3b1 = 1.0 if rng.random() < cfg.sf3b1_rate * (1.0 - 0.6 * bap1) else 0.0
        eif1ax = 1.0 if rng.random() < cfg.eif1ax_rate * (1.0 - 0.6 * bap1) else 0.0
        chr8q = 1.0 if rng.random() < float(np.clip(0.45 + 0.2 * bap1, 0.0, 1.0)) else 0.0

        class_logit = (
            1.9 * (bap1 - 0.5)
            + 1.3 * (mono - 0.5)
            + 0.5 * (chr8q - 0.5)
            + 0.15
            + float(rng.normal()) * 0.55
        )
        gep_class = 2 if rng.random() < _sigmoid(class_logit) else 1
        class2 = 1.0 if gep_class == 2 else 0.0

        eta = (
            1.15 * class2
            + 0.80 * bap1
            + 0.72 * mono
            + 0.35 * chr8q
            - 0.30 * sf3b1
            - 0.25 * eif1ax
            + 0.10 * gna11
        )
        risk = eta - _ETA_CENTER
        lam = float(np.exp(cfg.risk_scale * risk))
        t_event = float(rng.exponential(scale=1.0 / lam))
        t_censor = float(rng.exponential(scale=_CENSOR_SCALE))
        observed = min(t_event, t_censor)
        event = 1 if t_event <= t_censor else 0

        drivers = np.array([bap1, sf3b1, eif1ax, gnaq, gna11, mono, chr8q], dtype=np.float64)
        rna = (
            class2 * cfg.class_signal * v_class + 0.4 * bap1 * v_bap1 + rng.normal(size=cfg.n_genes)
        ).astype(np.float64)
        wsi: FloatArray | None
        if rng.random() < cfg.wsi_coverage:
            wsi = (
                bap1 * _MORPH_SCALE * EPITHELIOID_TEMPLATE + rng.normal(size=tile_dim) * 0.7
            ).astype(np.float64)
        else:
            wsi = None

        patients.append(
            Patient(
                pid=f"{name}-{idx:03d}",
                cohort=name,
                rna=rna,
                drivers=drivers,
                wsi=wsi,
                gep_class=gep_class,
                time=observed,
                event=event,
                latent=risk,
            )
        )
    return Cohort(name=name, patients=tuple(patients))


def build_dataset(cfg: CohortConfig) -> dict[str, Cohort]:
    out: dict[str, Cohort] = {}
    for offset, (name, size) in enumerate(cfg.sizes.items()):
        out[name] = generate_cohort(name, size, cfg, cfg.seed + 101 * offset)
    return out


def driver_frequencies(cohort: Cohort) -> NDArray[np.float64]:
    mat = np.stack([p.drivers for p in cohort.patients])
    return np.asarray(mat.mean(axis=0), dtype=np.float64)
