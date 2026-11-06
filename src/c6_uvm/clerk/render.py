from __future__ import annotations

from c6_uvm.bench.orchestrator import CaseOutcome
from c6_uvm.proceedings.pipeline import MainReport


def render_main(name: str, report: MainReport) -> str:
    lines: list[str] = []
    lines.append(f"experiment: {name}")
    lines.append("cohort      n    C6     95% CI          DecisionDx  dC      p        MCID")
    for row in report.rows:
        lines.append(
            f"{row.name:<11} {row.n:<4} {row.c6:.3f}  "
            f"[{row.c6_lo:.3f},{row.c6_hi:.3f}]  {row.decisiondx:.3f}      "
            f"{row.delta_c:+.3f}  {row.delong_p:.1e}  {'yes' if row.mcid_crossed else 'no'}"
        )
    lines.append(
        f"pooled C6   {report.pooled_c6:.3f} "
        f"[{report.pooled_lo:.3f},{report.pooled_hi:.3f}]  I2={report.i_squared:.0f}%  "
        f"p={report.pooled_p:.1e}"
    )
    lines.append(
        f"conformal   coverage={report.coverage:.3f} "
        f"band=[{report.band_lo:.3f},{report.band_hi:.3f}] refer={report.refer_rate:.3f} "
        f"(n_cal={report.n_cal})"
    )
    lines.append(f"calibration brier={report.brier:.3f} ece={report.ece:.3f}")
    nb = " ".join(f"{t:.2f}:{v:+.3f}" for t, v in report.net_benefit.items())
    lines.append(f"net-benefit {nb}")
    lines.append("baseline matrix (C-index)")
    cohorts = list(next(iter(report.baseline_matrix.values())).keys())
    header = "  method".ljust(24) + "".join(f"{c:>12}" for c in cohorts)
    lines.append(header)
    for method, by_cohort in report.baseline_matrix.items():
        cells = "".join(f"{by_cohort[c]:>12.3f}" for c in cohorts)
        lines.append("  " + method.ljust(22) + cells)
    return "\n".join(lines)


def render_case(outcome: CaseOutcome) -> str:
    lines: list[str] = []
    lines.append(f"case {outcome.pid}  GEP-class={outcome.gep_class}  risk={outcome.risk:+.3f}")
    lines.append(f"prediction set: [{outcome.lower:.3f}, {outcome.upper:.3f}]")
    lines.append(f"refer-for-expert-review: {'YES' if outcome.refer else 'no'}")
    lines.append(f"candidate therapies: {', '.join(outcome.drugs) if outcome.drugs else 'none'}")
    lines.append("evidence chain (KG-first order):")
    for source, node, score in outcome.trace:
        lines.append(f"  [{source:<10}] {node:<28} {score:+.3f}")
    return "\n".join(lines)
