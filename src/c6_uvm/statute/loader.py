from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from cattrs import Converter

from c6_uvm.statute.config import ExperimentConfig

_converter = Converter(forbid_extra_keys=True)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(text: str) -> Any:
    parsed = yaml.safe_load(text)
    return parsed


def _apply_override(tree: dict[str, Any], dotted: str) -> None:
    key, _, raw = dotted.partition("=")
    if not _:
        raise ValueError(f"override must be key=value, got {dotted!r}")
    cursor = tree
    parts = key.strip().split(".")
    for part in parts[:-1]:
        nxt = cursor.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"cannot descend into non-mapping at {part!r}")
        cursor = nxt
    cursor[parts[-1]] = _coerce(raw.strip())


def structure_config(tree: dict[str, Any]) -> ExperimentConfig:
    return _converter.structure(tree, ExperimentConfig)


def load_experiment(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    root = Path(path)
    raw = yaml.safe_load(root.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"experiment file {root} must be a mapping")
    bases = raw.pop("defaults", [])
    merged: dict[str, Any] = {}
    for rel in bases:
        sub = (root.parent / rel).with_suffix(".yaml")
        loaded = yaml.safe_load(sub.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"base file {sub} must be a mapping")
        merged = _deep_merge(merged, loaded)
    merged = _deep_merge(merged, raw)
    for dotted in overrides or []:
        _apply_override(merged, dotted)
    return structure_config(merged)


def dump_config(cfg: ExperimentConfig) -> dict[str, Any]:
    unstructured = _converter.unstructure(cfg)
    assert isinstance(unstructured, dict)
    return unstructured
