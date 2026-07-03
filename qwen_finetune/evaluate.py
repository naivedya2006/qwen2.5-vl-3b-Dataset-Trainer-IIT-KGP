from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_finetune.checkpoint import resolve_best_checkpoint
from qwen_finetune.collator import QwenVLInferenceCollator
from qwen_finetune.config import FrameworkConfig, apply_auto_profile
from qwen_finetune.dataset import DroneChatMLDataset
from qwen_finetune.logger import setup_logging
from qwen_finetune.metrics import generation_metrics
from qwen_finetune.model import load_base_model
from qwen_finetune.processor import load_processor
from qwen_finetune.utils import move_to_device, seed_everything, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and score Qwen2.5-VL predictions.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = FrameworkConfig.from_file(args.config) if args.config else FrameworkConfig()
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.max_samples is not None:
        cfg.data.max_validation_samples = args.max_samples
    cfg = apply_auto_profile(cfg)
    logger = setup_logging(cfg.training.output_dir)
    seed_everything(cfg.training.seed, cfg.training.deterministic, cfg.training.tf32)

    processor = load_processor(cfg.model, cfg.data)
    source = cfg.data.val_file if args.split == "val" else cfg.data.test_file
    dataset = DroneChatMLDataset(cfg.data, args.split, source)
    collator = QwenVLInferenceCollator(processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)

    adapter_path = args.adapter_path or resolve_best_checkpoint(cfg.training.output_dir)
    if not adapter_path:
        raise RuntimeError("No adapter checkpoint found. Pass --adapter-path or train first.")

    model = load_base_model(cfg.model)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    device = next(model.parameters()).device

    predictions: list[str] = []
    references: list[str] = []
    records: list[dict[str, str]] = []
    tokenizer = processor.tokenizer
    with torch.inference_mode():
        for batch in dataloader:
            inputs = move_to_device(batch["model_inputs"], device)
            generation_kwargs = {
                "max_new_tokens": cfg.generation.max_new_tokens,
                "do_sample": cfg.generation.do_sample,
                "num_beams": cfg.generation.num_beams,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if cfg.generation.do_sample:
                generation_kwargs["temperature"] = cfg.generation.temperature
                generation_kwargs["top_p"] = cfg.generation.top_p
            generated = model.generate(**inputs, **generation_kwargs)
            prompt_width = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
            predictions.extend(decoded)
            references.extend(batch["references"])
            for index, text in enumerate(decoded):
                records.append(
                    {
                        "sample_id": batch["sample_ids"][index],
                        "prompt": batch["prompts"][index],
                        "prediction": text,
                        "reference": batch["references"][index],
                        "video": batch["video_paths"][index],
                        "image": batch["image_paths"][index],
                    }
                )

    metrics = generation_metrics(predictions, references)
    output_path = Path(cfg.training.output_dir) / f"{args.split}_generations.json"
    metrics_path = Path(cfg.training.output_dir) / f"{args.split}_generation_metrics.json"
    write_json(output_path, records)
    write_json(metrics_path, metrics)
    logger.info("Wrote %s predictions to %s", len(records), output_path)
    logger.info("Generation metrics: %s", metrics)


if __name__ == "__main__":
    main()
