from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_finetune.checkpoint import resolve_resume_checkpoint, save_processor
from qwen_finetune.config import FrameworkConfig, apply_auto_profile
from qwen_finetune.dataset import build_datasets
from qwen_finetune.logger import setup_logging
from qwen_finetune.lora import apply_lora
from qwen_finetune.metrics import eval_loss_metrics
from qwen_finetune.model import configure_model_for_training, load_base_model
from qwen_finetune.processor import load_processor
from qwen_finetune.trainer import build_trainer
from qwen_finetune.utils import cuda_report, package_versions, seed_everything, write_json


def main(argv: list[str] | None = None) -> None:
    cfg = FrameworkConfig.from_args(argv)
    cfg = apply_auto_profile(cfg)
    logger = setup_logging(cfg.training.output_dir)
    seed_everything(cfg.training.seed, cfg.training.deterministic, cfg.training.tf32)
    cfg.save(Path(cfg.training.output_dir) / "resolved_config.json")

    logger.info("CUDA: %s", cuda_report())
    logger.info(
        "Packages: %s",
        package_versions(["torch", "transformers", "peft", "accelerate", "bitsandbytes", "opencv-python", "tensorboard"]),
    )
    logger.info("Resolved config written to %s", Path(cfg.training.output_dir) / "resolved_config.json")

    processor = load_processor(cfg.model, cfg.data)
    train_dataset, val_dataset, test_dataset = build_datasets(cfg.data)
    write_json(
        Path(cfg.training.output_dir) / "validation_report.json",
        {
            "train": train_dataset.report.to_dict(),
            "val": val_dataset.report.to_dict(),
            "test": test_dataset.report.to_dict(),
        },
    )

    if cfg.runtime.validate_only:
        logger.info("Validation-only run completed.")
        return

    if cfg.runtime.dry_run:
        from qwen_finetune.collator import QwenVLTrainingCollator

        collator = QwenVLTrainingCollator(processor)
        batch = collator([train_dataset[0], train_dataset[1]])
        logger.info(
            "Dry run batch ok: input_ids=%s assistant_label_tokens=%s",
            tuple(batch["input_ids"].shape),
            int((batch["labels"] != -100).sum().item()),
        )
        return

    model = load_base_model(cfg.model)
    model = configure_model_for_training(
        model,
        cfg.training.gradient_checkpointing,
        cfg.training.gradient_checkpointing_use_reentrant,
    )
    model = apply_lora(model, cfg.lora, cfg.training)
    trainer = build_trainer(cfg, model, processor, train_dataset, val_dataset)

    resume_checkpoint = resolve_resume_checkpoint(
        cfg.training.output_dir,
        cfg.training.resume_from_checkpoint,
    )
    if resume_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(cfg.training.output_dir)
    save_processor(processor, cfg.training.output_dir)

    eval_metrics = eval_loss_metrics(trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval"))
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    if cfg.runtime.run_test_eval:
        test_metrics = eval_loss_metrics(trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test"))
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
    logger.info("Training completed. Artifacts are in %s", cfg.training.output_dir)


if __name__ == "__main__":
    main()

