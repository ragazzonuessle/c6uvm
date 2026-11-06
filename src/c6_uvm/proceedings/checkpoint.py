from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    loaded = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint payload is not a mapping")
    return loaded
