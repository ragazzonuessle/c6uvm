from __future__ import annotations

import logging
from pathlib import Path

import typer

from c6_uvm.clerk.render import render_case, render_main
from c6_uvm.intake.cohort import build_dataset
from c6_uvm.proceedings.checkpoint import save_checkpoint
from c6_uvm.proceedings.pipeline import build_orchestrator, run_main
from c6_uvm.statute.loader import dump_config, load_experiment

app = typer.Typer(add_completion=False, help="C6 multi-agent uveal-melanoma decision pipeline.")

_DEFAULT = "configs/experiment/main.yaml"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("c6")


@app.command()
def train(
    experiment: str = typer.Option(_DEFAULT, "--experiment", "-e"),
    over: list[str] = typer.Option([], "--set", help="dotted overrides key=value"),
    seed: int | None = typer.Option(None, "--seed"),
    out: Path = typer.Option(Path("checkpoints/c6.pt"), "--out"),
) -> None:
    cfg = load_experiment(experiment, over)
    chosen = cfg.train.seed if seed is None else seed
    orchestrator = build_orchestrator(cfg, chosen)
    head = orchestrator.system.subtype_head
    payload = {
        "config": dump_config(cfg),
        "seed": chosen,
        "subtype_state": head.state_dict(),
        "router_backbone": orchestrator.system.router.backbone,
    }
    save_checkpoint(out, payload)
    _log.info("trained C6 (%s) seed=%d -> %s", cfg.name, chosen, out)


@app.command()
def evaluate(
    experiment: str = typer.Option(_DEFAULT, "--experiment", "-e"),
    over: list[str] = typer.Option([], "--set"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    cfg = load_experiment(experiment, over)
    report = run_main(cfg, seed)
    typer.echo(render_main(cfg.name, report))


@app.command()
def infer(
    experiment: str = typer.Option(_DEFAULT, "--experiment", "-e"),
    over: list[str] = typer.Option([], "--set"),
    seed: int | None = typer.Option(None, "--seed"),
    cohort: str = typer.Option("TCGA-UVM", "--cohort"),
    index: int = typer.Option(0, "--index"),
) -> None:
    cfg = load_experiment(experiment, over)
    chosen = cfg.train.seed if seed is None else seed
    orchestrator = build_orchestrator(cfg, chosen)
    data = build_dataset(cfg.cohort)
    if cohort not in data:
        raise typer.BadParameter(f"unknown cohort {cohort}")
    patients = data[cohort].patients
    patient = patients[index % len(patients)]
    outcome = orchestrator.run(patient)
    typer.echo(render_case(outcome))


@app.command()
def export(
    experiment: str = typer.Option(_DEFAULT, "--experiment", "-e"),
    over: list[str] = typer.Option([], "--set"),
    seed: int | None = typer.Option(None, "--seed"),
    out: Path = typer.Option(Path("artifacts/subtype.onnx"), "--out"),
) -> None:
    import torch

    cfg = load_experiment(experiment, over)
    chosen = cfg.train.seed if seed is None else seed
    orchestrator = build_orchestrator(cfg, chosen)
    head = orchestrator.system.subtype_head
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, cfg.model.rna_dim, dtype=torch.float32)
    torch.onnx.export(
        head,
        (dummy,),
        str(out),
        input_names=["rna"],
        output_names=["class2_logit"],
        dynamo=False,
    )
    _log.info("exported subtype head -> %s", out)


if __name__ == "__main__":
    app()
