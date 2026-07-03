from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

from .config import ModelConfig
from .logger import get_logger
from .utils import require_module, torch_dtype_from_name


def build_quantization_config(model_config: ModelConfig) -> BitsAndBytesConfig | None:
    if not model_config.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch_dtype_from_name(model_config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
    )


def select_attention_implementation(requested: str) -> str:
    requested = requested.lower()
    if requested in {"sdpa", "eager", "flash_attention_2"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported attention implementation: {requested}")
    if torch.cuda.is_available() and require_module("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


def load_base_model(model_config: ModelConfig) -> Any:
    logger = get_logger()
    if model_config.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit QLoRA requires a CUDA GPU.")

    dtype = torch_dtype_from_name(model_config.dtype)
    quantization_config = build_quantization_config(model_config)
    attn_implementation = select_attention_implementation(model_config.attn_implementation)
    kwargs: dict[str, Any] = {
        "revision": model_config.revision,
        "local_files_only": model_config.local_files_only,
        "trust_remote_code": model_config.trust_remote_code,
        "quantization_config": quantization_config,
        "device_map": {"": 0} if torch.cuda.is_available() else None,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    logger.info("Loading %s with %s attention and %s quantization", model_config.model_id, attn_implementation, "4-bit NF4" if quantization_config else "no")
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_config.model_id, **kwargs)
    except Exception as exc:
        if attn_implementation != "flash_attention_2":
            raise
        logger.warning("FlashAttention2 load failed, falling back to SDPA: %s", exc)
        kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForImageTextToText.from_pretrained(model_config.model_id, **kwargs)

    if not model_config.use_cache:
        disable_model_cache(model)
    return model


def configure_model_for_training(model: Any, gradient_checkpointing: bool, use_reentrant: bool) -> Any:
    disable_model_cache(model)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    return model


def disable_model_cache(model: Any) -> None:
    seen: set[int] = set()
    stack = [model]
    for attr in ("base_model", "model"):
        value = getattr(model, attr, None)
        if value is not None:
            stack.append(value)
    if hasattr(model, "modules"):
        stack.extend(list(model.modules()))
    for obj in stack:
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        config = getattr(obj, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
