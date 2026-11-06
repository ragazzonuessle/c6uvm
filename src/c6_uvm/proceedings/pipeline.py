from __future__ import annotations

import numpy as np
import torch
from attrs import frozen
from numpy.typing import NDArray

from c6_uvm.bench.orchestrator import Orchestrator
from c6_uvm.bench.system import C6System
from c6_uvm.findings.baselines import baseline_scores
from c6_uvm.findings.clinical import brier_score, expected_calibration_error, net_benefit
from c6_uvm.findings.concordance import bootstrap_ci, c_index, paired_concordance_test
from c6_uvm.findings.pooling import dersimonian_laird
from c6_uvm.intake.cohort import build_dataset
from c6_uvm.proceedings.split import pooled_split
from c6_uvm.record.seeding import set_seed
from c6_uvm.record.types import Cohort, Patient
from c6_uvm.ruling.conformal import SplitConformal
from c6_uvm.ruling.coverage import empirical_coverage, marginal_band
from c6_uvm.statute.config import ExperimentConfig

FloatVec = NDArray[np.float64]


@frozen
class CohortRow:
    name: str
    n: int
    c6: float
    c6_lo: float
    c6_hi: float
    decisiondx: float
    delta_c: float
    delong_p: float
    mcid_crossed: bool


@frozen
class MainReport:
    rows: tuple[CohortRow, ...]
    pooled_c6: float
    pooled_lo: float
    pooled_hi: float
    pooled_p: float
    i_squared: float
    baseline_matrix: dict[str, dict[str, float]]
    coverage: float
    band_lo: float
    band_hi: float
    refer_rate: float
    brier: float
    ece: float
    net_benefit: dict[float, float]
    n_cal: int


@frozen
class _Prepared:
    system: C6System
    conformal: SplitConformal
    data: dict[str, Cohort]
    train_pool: tuple[Patient, ...]
    n_cal: int
    seed: int


def _log_time(patients: tuple[Patient, ...] | list[Patient]) -> FloatVec:
    return np.array([np.log(p.time + 1e-6) for p in patients], dtype=np.float64)


def _platt(risk_train: FloatVec, event_train: NDArray[np.int64], risk_eval: FloatVec) -> FloatVec:
    xs = torch.as_tensor(risk_train, dtype=torch.float32)
    ys = torch.as_tensor(event_train.astype(np.float32), dtype=torch.float32)
    a = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(a * xs + b, ys)
        torch.autograd.backward(loss)
        opt.step()
    with torch.no_grad():
        probs = torch.sigmoid(a * torch.as_tensor(risk_eval, dtype=torch.float32) + b)
    return probs.numpy().astype(np.float64)


def _prepare(cfg: ExperimentConfig, seed: int) -> _Prepared:
    set_seed(seed)
    data = build_dataset(cfg.cohort)
    if cfg.holdout_cohort is not None:
        train_cohorts = [c for name, c in data.items() if name != cfg.holdout_cohort]
    else:
        train_cohorts = list(data.values())
    train_pool = tuple(p for cohort in train_cohorts for p in cohort.patients)

    system = C6System(cfg)
    system.fit(train_pool)

    split = pooled_split(list(train_pool), cfg.conformal.cal_fraction, seed + 7)
    risk_train = system.c6_risk(list(split.train))
    risk_cal = system.c6_risk(list(split.cal))
    conformal = SplitConformal(
        cfg.conformal.alpha, cfg.conformal.width_budget, cfg.conformal.pe_threshold
    )
    conformal.fit(
        risk_train,
        _log_time(split.train),
        risk_cal,
        _log_time(split.cal),
        cfg.conformal.quantile_steps,
        cfg.conformal.quantile_lr,
    )
    return _Prepared(
        system=system,
        conformal=conformal,
        data=data,
        train_pool=train_pool,
        n_cal=len(split.cal),
        seed=seed,
    )


def _cohort_row(name: str, cohort: Cohort, prep: _Prepared, cfg: ExperimentConfig) -> CohortRow:
    patients = list(cohort.patients)
    time = cohort.times()
    event = cohort.events()
    risk = prep.system.c6_risk(patients)
    point, lo, hi = bootstrap_ci(time, risk, event, cfg.eval.bootstrap, prep.seed + 3)
    drivers = np.stack([p.drivers for p in patients])
    gep = np.array([p.gep_class for p in patients], dtype=np.int64)
    dd_risk = baseline_scores(drivers, gep, prep.seed)["DecisionDx-UM GEP"]
    dd_c = c_index(time, dd_risk, event)
    p_value = paired_concordance_test(
        time, risk, dd_risk, event, cfg.eval.bootstrap, prep.seed + 11
    )
    delta = point - dd_c
    return CohortRow(
        name=name,
        n=len(patients),
        c6=point,
        c6_lo=lo,
        c6_hi=hi,
        decisiondx=dd_c,
        delta_c=delta,
        delong_p=p_value,
        mcid_crossed=delta >= cfg.eval.mcid,
    )


def _baseline_matrix(prep: _Prepared, cfg: ExperimentConfig) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for name, cohort in prep.data.items():
        patients = list(cohort.patients)
        time = cohort.times()
        event = cohort.events()
        drivers = np.stack([p.drivers for p in patients])
        gep = np.array([p.gep_class for p in patients], dtype=np.int64)
        scores = baseline_scores(drivers, gep, prep.seed)
        scores["Ours (C6)"] = prep.system.c6_risk(patients)
        for method, risk in scores.items():
            matrix.setdefault(method, {})[name] = c_index(time, risk, event)
    return matrix


def _conformal_block(prep: _Prepared, cfg: ExperimentConfig) -> tuple[float, float, float, float]:
    split = pooled_split(list(prep.train_pool), cfg.conformal.cal_fraction, prep.seed + 7)
    risk_test = prep.system.c6_risk(list(split.test))
    table = prep.system.score(list(split.test))
    sets = prep.conformal.predict_many(risk_test, table.pe_counts)
    coverage = empirical_coverage(sets, _log_time(split.test))
    refer_rate = float(np.mean([ps.refer for ps in sets])) if sets else 0.0
    band_lo, band_hi = marginal_band(prep.n_cal, cfg.conformal.alpha)
    return coverage, band_lo, band_hi, refer_rate


def _clinical_block(
    prep: _Prepared, cfg: ExperimentConfig
) -> tuple[float, float, dict[float, float]]:
    split = pooled_split(list(prep.train_pool), cfg.conformal.cal_fraction, prep.seed + 7)
    cal = list(split.cal)
    risk_train = prep.system.c6_risk(list(split.train))
    event_train = np.array([p.event for p in split.train], dtype=np.int64)
    risk_cal = prep.system.c6_risk(cal)
    outcome = np.array([p.event for p in cal], dtype=np.int64)
    prob = _platt(risk_train, event_train, risk_cal)
    brier = brier_score(prob, outcome)
    ece = expected_calibration_error(prob, outcome, cfg.eval.ece_bins)
    nb = {t: net_benefit(prob, outcome, t) for t in cfg.eval.net_benefit_thresholds}
    return brier, ece, nb


def _pooled_pvalue(prep: _Prepared, cfg: ExperimentConfig) -> float:
    patients = [p for cohort in prep.data.values() for p in cohort.patients]
    time = np.array([p.time for p in patients], dtype=np.float64)
    event = np.array([p.event for p in patients], dtype=np.int64)
    risk = prep.system.c6_risk(patients)
    drivers = np.stack([p.drivers for p in patients])
    gep = np.array([p.gep_class for p in patients], dtype=np.int64)
    dd_risk = baseline_scores(drivers, gep, prep.seed)["DecisionDx-UM GEP"]
    return paired_concordance_test(time, risk, dd_risk, event, cfg.eval.bootstrap, prep.seed + 23)


def run_main(cfg: ExperimentConfig, seed: int | None = None) -> MainReport:
    chosen = cfg.train.seed if seed is None else seed
    prep = _prepare(cfg, chosen)
    rows = tuple(
        _cohort_row(name, cohort, prep, cfg)
        for name, cohort in prep.data.items()
        if cfg.holdout_cohort is None or name == cfg.holdout_cohort
    )
    estimates = np.array([r.c6 for r in rows], dtype=np.float64)
    ses = np.array([max((r.c6_hi - r.c6_lo) / (2 * 1.96), 1e-3) for r in rows], dtype=np.float64)
    pooled = dersimonian_laird(estimates, ses**2)
    pooled_p = _pooled_pvalue(prep, cfg)
    matrix = _baseline_matrix(prep, cfg)
    coverage, band_lo, band_hi, refer_rate = _conformal_block(prep, cfg)
    brier, ece, nb = _clinical_block(prep, cfg)
    return MainReport(
        rows=rows,
        pooled_c6=pooled.estimate,
        pooled_lo=pooled.ci_low,
        pooled_hi=pooled.ci_high,
        pooled_p=pooled_p,
        i_squared=pooled.i_squared,
        baseline_matrix=matrix,
        coverage=coverage,
        band_lo=band_lo,
        band_hi=band_hi,
        refer_rate=refer_rate,
        brier=brier,
        ece=ece,
        net_benefit=nb,
        n_cal=prep.n_cal,
    )


def build_orchestrator(cfg: ExperimentConfig, seed: int | None = None) -> Orchestrator:
    chosen = cfg.train.seed if seed is None else seed
    prep = _prepare(cfg, chosen)
    return Orchestrator(prep.system, prep.conformal)
