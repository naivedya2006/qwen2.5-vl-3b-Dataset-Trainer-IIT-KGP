from __future__ import annotations

from typing import Any

from transformers import AutoProcessor

from .config import DataConfig, ModelConfig


def load_processor(model_config: ModelConfig, data_config: DataConfig) -> Any:
    max_pixels = max(data_config.image_max_pixels, data_config.video_max_pixels)
    processor = AutoProcessor.from_pretrained(
        model_config.model_id,
        revision=model_config.revision,
        local_files_only=model_config.local_files_only,
        trust_remote_code=model_config.trust_remote_code,
        min_pixels=data_config.min_pixels,
        max_pixels=max_pixels,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return processor


def assistant_marker_ids(processor: Any) -> list[int]:
    return processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)


def generation_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") != "assistant"]

