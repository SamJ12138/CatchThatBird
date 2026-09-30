from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Relative paths in config.yaml resolve against the project directory, never
# the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path: Path, root: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else root / path


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_every_n_frames: int = 30
    motion_min_area: int = 200
    motion_threshold: int = 16
    motion_padding_px: int = 50
    motion_warmup_frames: int = 60
    yolo_model: str = "yolov8n.pt"
    yolo_target_classes: list[str] = Field(default_factory=lambda: ["bird"])
    yolo_confidence_threshold: float = 0.35
    dedupe_within_seconds: int = 10


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events_file: Path = Path("data/events.jsonl")
    snapshots_dir: Path = Path("data/snapshots")
    snapshot_format: str = "jpeg"
    snapshot_quality: int = 90
    save_full_frame: bool = True


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_events_per_day: int = 500
    retention_days: int = 30


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: CameraConfig
    detection: DetectionConfig
    logging: LoggingConfig
    storage: StorageConfig


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
