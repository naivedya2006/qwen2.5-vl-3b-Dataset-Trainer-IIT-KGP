from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_finetune.checkpoint import resolve_best_checkpoint
from qwen_finetune.config import FrameworkConfig, apply_auto_profile
from qwen_finetune.logger import setup_logging
from qwen_finetune.model import load_base_model
from qwen_finetune.processor import load_processor
from qwen_finetune.utils import move_to_device, resolve_path, seed_everything
from qwen_finetune.video import VideoFrameSampler, load_image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Qwen2.5-VL drone pedestrian prediction.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Generate a road safety report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = FrameworkConfig.from_file(args.config) if args.config else FrameworkConfig()
    cfg = apply_auto_profile(cfg)
    logger = setup_logging(cfg.training.output_dir)
    seed_everything(cfg.training.seed, cfg.training.deterministic, cfg.training.tf32)

    processor = load_processor(cfg.model, cfg.data)
    root = Path(cfg.data.dataset_root).resolve()
    
    video_path = Path(args.video).resolve()
    image_path = Path(args.image).resolve()
    sampler = VideoFrameSampler(
        cfg.data.num_video_frames,
        cfg.data.video_max_pixels,
        cfg.data.frame_sample_strategy,
        cfg.data.video_cache_size,
    )
    frames = sampler.load(video_path)
    image = load_image(image_path, cfg.data.image_max_pixels)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], videos=[frames], padding=True, return_tensors="pt")

    adapter_path = args.adapter_path or resolve_best_checkpoint(cfg.training.output_dir)
    if not adapter_path:
        raise RuntimeError("No adapter checkpoint found. Pass --adapter-path or train first.")
    model = load_base_model(cfg.model)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    device = next(model.parameters()).device
    inputs = move_to_device(dict(inputs), device)
    generation_kwargs = {
        "max_new_tokens": cfg.generation.max_new_tokens,
        "do_sample": cfg.generation.do_sample,
        "num_beams": cfg.generation.num_beams,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if cfg.generation.do_sample:
        generation_kwargs["temperature"] = cfg.generation.temperature
        generation_kwargs["top_p"] = cfg.generation.top_p
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    answer = processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    logger.info("Prediction:\n%s", answer.strip())
    print(answer.strip())


if __name__ == "__main__":
    main()
