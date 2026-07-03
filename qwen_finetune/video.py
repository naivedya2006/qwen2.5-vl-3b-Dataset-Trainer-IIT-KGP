from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float


class LruCache:
    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(0, max_entries)
        self._items: OrderedDict[object, object] = OrderedDict()

    def get(self, key: object) -> object | None:
        if self.max_entries <= 0 or key not in self._items:
            return None
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, key: object, value: object) -> None:
        if self.max_entries <= 0:
            return
        if key in self._items:
            self._items.pop(key)
        self._items[key] = value
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)


def resize_to_pixel_budget(image: Image.Image, max_pixels: int) -> Image.Image:
    if max_pixels <= 0:
        return image
    width, height = image.size
    pixels = width * height
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / float(pixels))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = image.copy()
    resized.thumbnail(new_size, Image.Resampling.BICUBIC)
    return resized


def load_image(path: str | Path, max_pixels: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.load()
    return resize_to_pixel_budget(rgb, max_pixels)


def validate_image(path: str | Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True, None
    except Exception as exc:
        return False, f"invalid image: {exc}"


def probe_video(path: str | Path, read_first_frame: bool = False) -> VideoMetadata:
    path = Path(path)
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open video")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if read_first_frame:
            ok, _ = cap.read()
            if not ok:
                raise ValueError("OpenCV could not decode first frame")
        if frame_count <= 0:
            raise ValueError("video reports zero frames")
        if width <= 0 or height <= 0:
            raise ValueError("video reports invalid dimensions")
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(
            path=str(path),
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=duration,
        )
    finally:
        cap.release()


def validate_video(path: str | Path) -> tuple[bool, str | None]:
    try:
        probe_video(path, read_first_frame=True)
        return True, None
    except Exception as exc:
        return False, f"invalid video: {exc}"


def sample_frame_indices(frame_count: int, num_frames: int, strategy: str = "uniform") -> list[int]:
    if frame_count <= 0:
        return []
    num_frames = max(1, min(num_frames, frame_count))
    if num_frames == 1:
        return [frame_count // 2]
    if strategy == "head_tail":
        return sorted({0, frame_count - 1, *(round(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames))})
    return [round(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames)]


class VideoFrameSampler:
    """Decode only selected video frames, never whole clips."""

    def __init__(
        self,
        num_frames: int,
        max_pixels: int,
        strategy: str = "uniform",
        cache_size: int = 48,
    ) -> None:
        self.num_frames = num_frames
        self.max_pixels = max_pixels
        self.strategy = strategy
        self._metadata_cache = LruCache(max(cache_size, 8))
        self._frame_cache = LruCache(cache_size)

    def metadata(self, path: str | Path) -> VideoMetadata:
        path = str(Path(path))
        cached = self._metadata_cache.get(path)
        if cached is not None:
            return cached  # type: ignore[return-value]
        metadata = probe_video(path, read_first_frame=False)
        self._metadata_cache.put(path, metadata)
        return metadata

    def load(self, path: str | Path) -> list[Image.Image]:
        metadata = self.metadata(path)
        indices = sample_frame_indices(metadata.frame_count, self.num_frames, self.strategy)
        key = (metadata.path, tuple(indices), self.max_pixels)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return [frame.copy() for frame in cached]  # type: ignore[arg-type]

        frames = self._decode_opencv(metadata.path, indices)
        if not frames:
            raise ValueError(f"Could not decode sampled frames from {metadata.path}")
        while len(frames) < len(indices):
            frames.append(frames[-1].copy())
        frames = [resize_to_pixel_budget(frame, self.max_pixels) for frame in frames[: len(indices)]]
        self._frame_cache.put(key, [frame.copy() for frame in frames])
        return frames

    def _decode_opencv(self, path: str, indices: Iterable[int]) -> list[Image.Image]:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        frames: list[Image.Image] = []
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            for index in indices:
                decoded = self._read_exact_frame(cap, index, fps)
                if decoded is not None:
                    frames.append(decoded)
        finally:
            cap.release()
        return frames

    @staticmethod
    def _read_exact_frame(cap: cv2.VideoCapture, index: int, fps: float) -> Image.Image | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
        ok, frame = cap.read()
        if not ok and fps > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, index / fps * 1000.0))
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

