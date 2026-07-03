from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_finetune.collator import QwenVLTrainingCollator
from qwen_finetune.config import FrameworkConfig, apply_auto_profile
from qwen_finetune.dataset import build_datasets
from qwen_finetune.logger import setup_logging
from qwen_finetune.lora import apply_lora
from qwen_finetune.model import configure_model_for_training, load_base_model
from qwen_finetune.processor import load_processor
from qwen_finetune.utils import cuda_report, move_to_device, package_versions, seed_everything, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run framework self-audit checks.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--load-model", action="store_true", help="Also load Qwen weights and run a one-batch grad check.")
    parser.add_argument("--max-validation-samples", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = FrameworkConfig.from_file(args.config) if args.config else FrameworkConfig()
    if args.max_validation_samples is not None:
        cfg.data.max_validation_samples = args.max_validation_samples
    cfg = apply_auto_profile(cfg)
    logger = setup_logging(cfg.training.output_dir)
    seed_everything(cfg.training.seed, cfg.training.deterministic, cfg.training.tf32)

    versions = package_versions(
        ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "opencv-python", "torchvision", "av", "tensorboard"]
    )
    logger.info("CUDA: %s", cuda_report())
    logger.info("Packages: %s", versions)
    processor = load_processor(cfg.model, cfg.data)
    train, val, test = build_datasets(cfg.data)
    collator = QwenVLTrainingCollator(processor)
    batch = collator([train[0], train[1]])
    labels = batch["labels"]
    if int((labels != -100).sum().item()) <= 0:
        raise AssertionError("Label mask contains no trainable assistant tokens")
    if "pixel_values_videos" not in batch:
        raise AssertionError("Processor did not produce video tensors")
    if "pixel_values" not in batch:
        raise AssertionError("Processor did not produce image tensors")

    report = {
        "cuda": cuda_report(),
        "packages": versions,
        "config": cfg.to_dict(),
        "dataset": {
            "train": train.report.to_dict(),
            "val": val.report.to_dict(),
            "test": test.report.to_dict(),
        },
        "collator": {
            "input_ids_shape": list(batch["input_ids"].shape),
            "assistant_label_tokens": int((labels != -100).sum().item()),
            "has_video": "pixel_values_videos" in batch,
            "has_image": "pixel_values" in batch,
        },
    }

    if args.load_model:
        model = load_base_model(cfg.model)
        model = configure_model_for_training(
            model,
            cfg.training.gradient_checkpointing,
            cfg.training.gradient_checkpointing_use_reentrant,
        )
        model = apply_lora(model, cfg.lora, cfg.training)
        model.train()
        device = next(model.parameters()).device
        model_batch = move_to_device(batch, device)
        out = model(**model_batch)
        out.loss.backward()
        lora_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.requires_grad
        ]
        if not any(grad is not None and torch.isfinite(grad).all() for grad in lora_grads):
            raise AssertionError("No finite LoRA gradients found after backward")
        report["gradient_check"] = {"loss": float(out.loss.detach().cpu()), "lora_grads": len(lora_grads)}

    audit_path = Path(cfg.training.output_dir) / "audit_report.json"
    write_json(audit_path, report)
    logger.info("Audit passed. Report: %s", audit_path)


if __name__ == "__main__":
    main()

