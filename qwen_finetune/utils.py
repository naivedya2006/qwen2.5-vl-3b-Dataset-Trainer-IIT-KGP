from __future__ import annotations

import importlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_path(root: str | os.PathLike[str], value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


def require_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[normalized]


def seed_everything(seed: int, deterministic: bool = False, tf32: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def cuda_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.update(
            {
                "device": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return report


def package_versions(packages: list[str]) -> dict[str, str]:
    try:
        from importlib import metadata
    except ImportError:
        import importlib_metadata as metadata  # type: ignore

    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except Exception:
            versions[package] = "missing"
    return versions


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def format_parameter_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def move_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def latest_checkpoint_from_trainer_state(output_dir: str | os.PathLike[str]) -> str | None:
    state_path = Path(output_dir) / "trainer_state.json"
    if not state_path.exists():
        return None
    try:
        state = read_json(state_path)
    except Exception:
        return None
    best = state.get("best_model_checkpoint")
    if best:
        return str(best)
    return None


_SEVERITY_RE = re.compile(r"(?:final\s+severity|severity(?:\s+score)?)\s*[:=]?\s*([0-9]+)", re.I)


def extract_severity(text: str) -> int | None:
    match = _SEVERITY_RE.search(text or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def safe_perplexity(loss: float) -> float:
    if not math.isfinite(loss):
        return float("nan")
    return float(math.exp(min(loss, 20.0)))
