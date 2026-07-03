from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from .checkpoint import save_processor
from .collator import QwenVLTrainingCollator
from .config import FrameworkConfig
from .logger import get_logger


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            save_processor(self.processor, checkpoint_dir)
        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        save_processor(self.processor, args.output_dir)
        return control


class MemoryLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
        if logs is not None and torch.cuda.is_available():
            logs["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
            logs["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
        return control


def build_training_arguments(cfg: FrameworkConfig) -> TrainingArguments:
    training = cfg.training
    params = inspect.signature(TrainingArguments.__init__).parameters
    os.environ.setdefault(
        "TENSORBOARD_LOGGING_DIR",
        str(Path(training.output_dir) / "tensorboard"),
    )
    kwargs: dict[str, Any] = {
        "output_dir": training.output_dir,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "num_train_epochs": training.num_train_epochs,
        "max_steps": training.max_steps,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "warmup_steps": training.warmup_steps,
        "lr_scheduler_type": training.lr_scheduler_type,
        "optim": training.optim,
        "max_grad_norm": training.max_grad_norm,
        "bf16": training.bf16,
        "fp16": training.fp16,
        "tf32": training.tf32,
        "gradient_checkpointing": training.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": training.gradient_checkpointing_use_reentrant
        },
        "logging_strategy": "steps",
        "logging_steps": training.logging_steps,
        "logging_first_step": training.logging_first_step,
        "save_strategy": "steps",
        "save_steps": training.save_steps,
        "save_total_limit": training.save_total_limit,
        "load_best_model_at_end": training.load_best_model_at_end,
        "metric_for_best_model": training.metric_for_best_model,
        "greater_is_better": training.greater_is_better,
        "report_to": training.report_to,
        "run_name": Path(training.output_dir).name,
        "seed": training.seed,
        "data_seed": training.seed,
        "full_determinism": training.deterministic,
        "auto_find_batch_size": training.auto_find_batch_size,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "dataloader_num_workers": training.dataloader_num_workers,
        "dataloader_pin_memory": training.dataloader_pin_memory,
        "dataloader_persistent_workers": training.dataloader_persistent_workers
        and training.dataloader_num_workers > 0,
        "torch_empty_cache_steps": training.torch_empty_cache_steps,
        "eval_on_start": training.eval_on_start,
        "save_only_model": training.save_only_model,
        "do_train": True,
        "do_eval": True,
        "use_cache": False,
    }
    if training.dataloader_num_workers > 0 and training.dataloader_prefetch_factor > 0:
        kwargs["dataloader_prefetch_factor"] = training.dataloader_prefetch_factor
    if "eval_strategy" not in params:
        raise RuntimeError("This framework requires a current Transformers build with TrainingArguments.eval_strategy.")
    kwargs["eval_strategy"] = "steps"
    kwargs["eval_steps"] = training.eval_steps
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in params})


def build_trainer(
    cfg: FrameworkConfig,
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any,
) -> Trainer:
    logger = get_logger()
    args = build_training_arguments(cfg)
    collator = QwenVLTrainingCollator(processor)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "callbacks": [SaveProcessorCallback(processor), MemoryLoggingCallback()],
    }
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = processor
    logger.info(
        "Trainer: batch=%s grad_accum=%s workers=%s eval_steps=%s save_steps=%s",
        cfg.training.per_device_train_batch_size,
        cfg.training.gradient_accumulation_steps,
        cfg.training.dataloader_num_workers,
        cfg.training.eval_steps,
        cfg.training.save_steps,
    )
    return Trainer(**trainer_kwargs)
