from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers.trainer_utils import get_last_checkpoint

from .utils import latest_checkpoint_from_trainer_state


def resolve_resume_checkpoint(output_dir: str, requested: str | None) -> str | None:
    if not requested or requested.lower() in {"none", "false", "0"}:
        return None
    if requested.lower() != "auto":
        path = Path(requested)
        return str(path) if path.exists() else None
    output = Path(output_dir)
    if not output.exists():
        return None
    return get_last_checkpoint(str(output))


def resolve_best_checkpoint(output_dir: str) -> str | None:
    best = latest_checkpoint_from_trainer_state(output_dir)
    if best:
        return best
    return get_last_checkpoint(output_dir)


def save_processor(processor: Any, output_dir: str | Path) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(output_dir)

