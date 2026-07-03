from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


@dataclass(slots=True)
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    revision: str = "main"
    local_files_only: bool = False
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    attn_implementation: str = "auto"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    use_cache: bool = False


@dataclass(slots=True)
class DataConfig:
    dataset_root: str = "Qwen2.5VL_Dataset"
    train_file: str = "chatml_train.json"
    val_file: str = "chatml_val.json"
    test_file: str = "chatml_test.json"
    validate_media: bool = True
    skip_invalid_samples: bool = True
    max_validation_samples: int | None = None
    runtime_decode_retries: int = 32
    frame_sample_strategy: str = "uniform"
    num_video_frames: int = 0
    min_video_frames: int = 2
    max_video_frames: int = 6
    image_max_pixels: int = 0
    video_max_pixels: int = 0
    min_pixels: int = 50_176
    video_cache_size: int = 48
    image_cache_size: int = 128


@dataclass(slots=True)
class LoraTuningConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    use_rslora: bool = True
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    tune_vision: bool = False
    vision_target_modules: list[str] = field(
        default_factory=lambda: ["qkv", "proj", "fc1", "fc2"]
    )


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "outputs/qwen2_5_vl_3b_drone_qlora"
    num_train_epochs: float = 3.0
    max_steps: int = -1
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_steps: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    per_device_train_batch_size: int = 0
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 0
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    logging_first_step: bool = True
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    eval_on_start: bool = False
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    seed: int = 3407
    deterministic: bool = False
    auto_find_batch_size: bool = False
    dataloader_num_workers: int = -1
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    torch_empty_cache_steps: int = 50
    resume_from_checkpoint: str = "auto"
    save_only_model: bool = False


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9
    num_beams: int = 1


@dataclass(slots=True)
class RuntimeConfig:
    dry_run: bool = False
    validate_only: bool = False
    run_test_eval: bool = True


@dataclass(slots=True)
class FrameworkConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraTuningConfig = field(default_factory=LoraTuningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "FrameworkConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FrameworkConfig":
        cfg = cls()
        for section_name, section_value in raw.items():
            if not hasattr(cfg, section_name):
                raise ValueError(f"Unknown config section: {section_name}")
            section = getattr(cfg, section_name)
            for key, value in section_value.items():
                if not hasattr(section, key):
                    raise ValueError(f"Unknown config key: {section_name}.{key}")
                setattr(section, key, value)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | os.PathLike[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> "FrameworkConfig":
        parser = argparse.ArgumentParser(
            description="Fine-tune Qwen2.5-VL-3B-Instruct on the drone pedestrian ChatML dataset."
        )
        parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
        parser.add_argument("--dataset-root", type=str, default=None)
        parser.add_argument("--model-id", type=str, default=None)
        parser.add_argument("--output-dir", type=str, default=None)
        parser.add_argument("--epochs", type=float, default=None)
        parser.add_argument("--max-steps", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--grad-accum", type=int, default=None)
        parser.add_argument("--workers", type=int, default=None)
        parser.add_argument("--num-video-frames", type=int, default=None)
        parser.add_argument("--image-max-pixels", type=int, default=None)
        parser.add_argument("--video-max-pixels", type=int, default=None)
        parser.add_argument("--eval-steps", type=int, default=None)
        parser.add_argument("--save-steps", type=int, default=None)
        parser.add_argument("--resume-from-checkpoint", type=str, default=None)
        parser.add_argument("--local-files-only", action="store_true")
        parser.add_argument("--no-media-validation", action="store_true")
        parser.add_argument("--deterministic", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--validate-only", action="store_true")
        parser.add_argument("--no-test-eval", action="store_true")
        parser.add_argument("--max-validation-samples", type=int, default=None)
        args = parser.parse_args(argv)

        cfg = cls.from_file(args.config) if args.config else cls()
        if args.dataset_root is not None:
            cfg.data.dataset_root = args.dataset_root
        if args.model_id is not None:
            cfg.model.model_id = args.model_id
        if args.output_dir is not None:
            cfg.training.output_dir = args.output_dir
        if args.epochs is not None:
            cfg.training.num_train_epochs = args.epochs
        if args.max_steps is not None:
            cfg.training.max_steps = args.max_steps
        if args.learning_rate is not None:
            cfg.training.learning_rate = args.learning_rate
        if args.batch_size is not None:
            cfg.training.per_device_train_batch_size = args.batch_size
        if args.grad_accum is not None:
            cfg.training.gradient_accumulation_steps = args.grad_accum
        if args.workers is not None:
            cfg.training.dataloader_num_workers = args.workers
        if args.num_video_frames is not None:
            cfg.data.num_video_frames = args.num_video_frames
        if args.image_max_pixels is not None:
            cfg.data.image_max_pixels = args.image_max_pixels
        if args.video_max_pixels is not None:
            cfg.data.video_max_pixels = args.video_max_pixels
        if args.eval_steps is not None:
            cfg.training.eval_steps = args.eval_steps
        if args.save_steps is not None:
            cfg.training.save_steps = args.save_steps
        if args.resume_from_checkpoint is not None:
            cfg.training.resume_from_checkpoint = args.resume_from_checkpoint
        if args.local_files_only:
            cfg.model.local_files_only = True
        if args.no_media_validation:
            cfg.data.validate_media = False
        if args.deterministic:
            cfg.training.deterministic = True
        if args.dry_run:
            cfg.runtime.dry_run = True
        if args.validate_only:
            cfg.runtime.validate_only = True
        if args.no_test_eval:
            cfg.runtime.run_test_eval = False
        if args.max_validation_samples is not None:
            cfg.data.max_validation_samples = args.max_validation_samples
        return cfg


def apply_auto_profile(cfg: FrameworkConfig) -> FrameworkConfig:
    """Fill hardware-sensitive automatic fields for the local CUDA device."""
    memory_gb = _cuda_total_memory_gb()

    if cfg.data.num_video_frames <= 0:
        if memory_gb and memory_gb < 10:
            cfg.data.num_video_frames = max(cfg.data.min_video_frames, 2)
        elif memory_gb and memory_gb < 13:
            cfg.data.num_video_frames = 4
        else:
            cfg.data.num_video_frames = min(cfg.data.max_video_frames, 6)

    if cfg.data.image_max_pixels <= 0:
        cfg.data.image_max_pixels = 262_144 if not memory_gb or memory_gb >= 11 else 200_704
    if cfg.data.video_max_pixels <= 0:
        cfg.data.video_max_pixels = 200_704 if memory_gb and memory_gb < 13 else 262_144

    if cfg.training.per_device_train_batch_size <= 0:
        cfg.training.per_device_train_batch_size = 1
    if cfg.training.gradient_accumulation_steps <= 0:
        cfg.training.gradient_accumulation_steps = 8 if not memory_gb or memory_gb >= 11 else 16

    if cfg.training.dataloader_num_workers < 0:
        cfg.training.dataloader_num_workers = 2 if os.name == "nt" else 4
    if cfg.training.dataloader_num_workers == 0:
        cfg.training.dataloader_persistent_workers = False
        cfg.training.dataloader_prefetch_factor = 0

    cfg.data.num_video_frames = max(
        cfg.data.min_video_frames,
        min(cfg.data.num_video_frames, cfg.data.max_video_frames),
    )
    return cfg


def _cuda_total_memory_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return None
