from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from .config import DataConfig
from .logger import get_logger
from .utils import read_json, resolve_path
from .video import VideoFrameSampler, load_image, validate_image, validate_video


@dataclass(frozen=True, slots=True)
class ChatMLSample:
    sample_id: str
    index: int
    messages: list[dict[str, Any]]
    video_relpath: str
    image_relpath: str
    video_path: str
    image_path: str
    user_text: str
    assistant_text: str


@dataclass(slots=True)
class LoadedSample:
    sample_id: str
    messages: list[dict[str, Any]]
    image: Image.Image
    video_frames: list[Image.Image]
    user_text: str
    assistant_text: str
    video_path: str
    image_path: str


@dataclass(slots=True)
class ValidationReport:
    split: str
    source_file: str
    total: int
    valid: int
    skipped: int
    reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source_file": self.source_file,
            "total": self.total,
            "valid": self.valid,
            "skipped": self.skipped,
            "reasons": self.reasons,
        }


class DroneChatMLDataset(Dataset[LoadedSample]):
    def __init__(
        self,
        data_config: DataConfig,
        split: str,
        source_file: str,
    ) -> None:
        self.data_config = data_config
        self.split = split
        self.root = Path(data_config.dataset_root).resolve()
        self.source_path = resolve_path(self.root, source_file)
        self.video_sampler = VideoFrameSampler(
            num_frames=data_config.num_video_frames,
            max_pixels=data_config.video_max_pixels,
            strategy=data_config.frame_sample_strategy,
            cache_size=data_config.video_cache_size,
        )
        self._image_cache = {}
        self.samples, self.report = self._load_and_validate()
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {self.source_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> LoadedSample:
        errors: list[str] = []
        for offset in range(max(1, self.data_config.runtime_decode_retries)):
            sample = self.samples[(index + offset) % len(self.samples)]
            try:
                return self._load_sample(sample)
            except Exception as exc:
                errors.append(f"{sample.sample_id}: {exc}")
                continue
        raise RuntimeError(
            f"Could not load media for sample index {index} after recovery attempts: {errors[:3]}"
        )

    def _load_sample(self, sample: ChatMLSample) -> LoadedSample:
        image = self._load_cached_image(sample.image_path)
        video_frames = self.video_sampler.load(sample.video_path)
        return LoadedSample(
            sample_id=sample.sample_id,
            messages=sample.messages,
            image=image,
            video_frames=video_frames,
            user_text=sample.user_text,
            assistant_text=sample.assistant_text,
            video_path=sample.video_path,
            image_path=sample.image_path,
        )

    def _load_cached_image(self, path: str) -> Image.Image:
        key = (path, self.data_config.image_max_pixels)
        cached = self._image_cache.get(key)
        if cached is not None:
            return cached.copy()
        image = load_image(path, self.data_config.image_max_pixels)
        if len(self._image_cache) >= self.data_config.image_cache_size:
            self._image_cache.pop(next(iter(self._image_cache)))
        self._image_cache[key] = image.copy()
        return image

    def _load_and_validate(self) -> tuple[list[ChatMLSample], ValidationReport]:
        logger = get_logger()
        raw = read_json(self.source_path)
        if not isinstance(raw, list):
            raise ValueError(f"{self.source_path} must contain a JSON list")
        limit = self.data_config.max_validation_samples
        iterable = raw if limit is None else raw[:limit]
        samples: list[ChatMLSample] = []
        reasons: Counter[str] = Counter()
        media_status: dict[str, tuple[bool, str | None]] = {}

        for index, item in enumerate(iterable):
            parsed, reason = self._parse_sample(item, index)
            if parsed is None:
                reasons[reason or "invalid sample"] += 1
                continue

            media_ok = True
            for media_path, kind in ((parsed.video_path, "video"), (parsed.image_path, "image")):
                if media_path not in media_status:
                    if not Path(media_path).exists():
                        media_status[media_path] = (False, f"missing {kind}")
                    elif self.data_config.validate_media and kind == "video":
                        media_status[media_path] = validate_video(media_path)
                    elif self.data_config.validate_media and kind == "image":
                        media_status[media_path] = validate_image(media_path)
                    else:
                        media_status[media_path] = (True, None)
                ok, media_reason = media_status[media_path]
                if not ok:
                    media_ok = False
                    reasons[media_reason or f"invalid {kind}"] += 1
                    break
            if media_ok:
                samples.append(parsed)

        report = ValidationReport(
            split=self.split,
            source_file=str(self.source_path),
            total=len(iterable),
            valid=len(samples),
            skipped=len(iterable) - len(samples),
            reasons=dict(reasons),
        )
        logger.info(
            "Validated %s: %s valid, %s skipped from %s",
            self.split,
            report.valid,
            report.skipped,
            self.source_path.name,
        )
        if report.skipped:
            logger.warning("Skipped %s samples in %s: %s", report.skipped, self.split, report.reasons)
            if not self.data_config.skip_invalid_samples:
                raise RuntimeError(f"Invalid samples found in {self.source_path}: {report.reasons}")
        return samples, report

    def _parse_sample(self, item: Any, index: int) -> tuple[ChatMLSample | None, str | None]:
        if not isinstance(item, dict):
            return None, "sample is not an object"
        if item.get("type") != "chatml":
            return None, "sample type is not chatml"
        messages = item.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            return None, "messages missing or too short"
        if messages[0].get("role") != "user":
            return None, "first message is not user"
        if messages[-1].get("role") != "assistant":
            return None, "last message is not assistant"

        user_content = messages[0].get("content")
        assistant_content = messages[-1].get("content")
        if not isinstance(user_content, list) or not isinstance(assistant_content, list):
            return None, "content is not a list"

        video_relpath = _single_content_value(user_content, "video", "video")
        image_relpath = _single_content_value(user_content, "image", "image")
        user_text = _joined_text(user_content)
        assistant_text = _joined_text(assistant_content)
        if not video_relpath:
            return None, "video path missing"
        if not image_relpath:
            return None, "image path missing"
        if not user_text:
            return None, "user text missing"
        if not assistant_text:
            return None, "assistant text missing"

        video_path = resolve_path(self.root, video_relpath)
        image_path = resolve_path(self.root, image_relpath)
        sample_id = f"{self.split}-{index:06d}"
        return (
            ChatMLSample(
                sample_id=sample_id,
                index=index,
                messages=messages,
                video_relpath=video_relpath,
                image_relpath=image_relpath,
                video_path=str(video_path),
                image_path=str(image_path),
                user_text=user_text,
                assistant_text=assistant_text,
            ),
            None,
        )


def _single_content_value(content: list[dict[str, Any]], item_type: str, key: str) -> str | None:
    values = [item.get(key) for item in content if isinstance(item, dict) and item.get("type") == item_type]
    values = [value for value in values if isinstance(value, str) and value.strip()]
    return values[0] if len(values) == 1 else None


def _joined_text(content: list[dict[str, Any]]) -> str:
    parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
    return "\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip()).strip()


def build_datasets(data_config: DataConfig) -> tuple[DroneChatMLDataset, DroneChatMLDataset, DroneChatMLDataset]:
    train = DroneChatMLDataset(data_config, "train", data_config.train_file)
    val = DroneChatMLDataset(data_config, "val", data_config.val_file)
    test = DroneChatMLDataset(data_config, "test", data_config.test_file)
    return train, val, test
