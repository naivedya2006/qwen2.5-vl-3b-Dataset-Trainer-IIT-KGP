from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .dataset import LoadedSample
from .processor import assistant_marker_ids, generation_messages


def _find_subsequence(sequence: list[int], pattern: list[int]) -> int:
    if not pattern or len(sequence) < len(pattern):
        return -1
    last = -1
    end = len(sequence) - len(pattern) + 1
    for index in range(end):
        if sequence[index : index + len(pattern)] == pattern:
            last = index
    return last


@dataclass(slots=True)
class QwenVLTrainingCollator:
    processor: Any
    ignore_index: int = -100
    _assistant_marker: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self._assistant_marker = assistant_marker_ids(self.processor)

    def __call__(self, features: list[LoadedSample]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Received an empty batch")

        texts = [
            self.processor.apply_chat_template(
                sample.messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for sample in features
        ]
        images = [sample.image for sample in features]
        videos = [sample.video_frames for sample in features]

        batch = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        attention_mask = batch.get("attention_mask")
        for row in range(input_ids.size(0)):
            ids = input_ids[row].tolist()
            marker_start = _find_subsequence(ids, self._assistant_marker)
            if marker_start < 0:
                labels[row, :] = self.ignore_index
                continue
            answer_start = marker_start + len(self._assistant_marker)
            labels[row, :answer_start] = self.ignore_index
            if attention_mask is not None:
                labels[row, attention_mask[row] == 0] = self.ignore_index

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = self.ignore_index
        batch["labels"] = labels
        return dict(batch)


@dataclass(slots=True)
class QwenVLInferenceCollator:
    processor: Any

    def __call__(self, features: list[LoadedSample]) -> dict[str, Any]:
        texts = [
            self.processor.apply_chat_template(
                generation_messages(sample.messages),
                tokenize=False,
                add_generation_prompt=True,
            )
            for sample in features
        ]
        images = [sample.image for sample in features]
        videos = [sample.video_frames for sample in features]
        batch = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        return {
            "model_inputs": dict(batch),
            "sample_ids": [sample.sample_id for sample in features],
            "references": [sample.assistant_text for sample in features],
            "prompts": [sample.user_text for sample in features],
            "video_paths": [sample.video_path for sample in features],
            "image_paths": [sample.image_path for sample in features],
        }

