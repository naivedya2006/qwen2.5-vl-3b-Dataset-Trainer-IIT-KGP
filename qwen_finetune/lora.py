from __future__ import annotations

from typing import Any

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from .config import LoraTuningConfig, TrainingConfig
from .logger import get_logger
from .model import disable_model_cache
from .utils import count_trainable_parameters, format_parameter_count


def apply_lora(model: Any, lora_config: LoraTuningConfig, training_config: TrainingConfig) -> Any:
    logger = get_logger()
    gradient_kwargs = {"use_reentrant": training_config.gradient_checkpointing_use_reentrant}
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=training_config.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_kwargs,
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = list(lora_config.target_modules)
    if lora_config.tune_vision:
        target_modules.extend(lora_config.vision_target_modules)
    target_modules = _existing_target_modules(model, target_modules)
    if not target_modules:
        raise RuntimeError("No LoRA target modules matched the loaded model.")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        bias=lora_config.bias,
        use_rslora=lora_config.use_rslora,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    disable_model_cache(model)
    trainable, total = count_trainable_parameters(model)
    logger.info(
        "LoRA targets: %s | trainable %s / %s parameters (%.4f%%)",
        target_modules,
        format_parameter_count(trainable),
        format_parameter_count(total),
        100.0 * trainable / max(total, 1),
    )
    return model


def _existing_target_modules(model: Any, candidates: list[str]) -> list[str]:
    module_names = [name for name, _ in model.named_modules()]
    matched: list[str] = []
    for candidate in candidates:
        suffix = f".{candidate}"
        if any(name == candidate or name.endswith(suffix) for name in module_names):
            matched.append(candidate)
    return matched
