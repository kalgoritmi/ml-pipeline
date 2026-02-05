"""Configuration schema using Pydantic."""

import logging
import urllib.request
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class ColumnWeight(BaseModel):
    name: str
    weight: float


class OperationConfig(BaseModel):
    type: Literal["IndexOperation", "RemoveColumns", "ComputeTarget", "Shuffle", "LimitRows"]
    column: str | None = None
    columns: list[str] | list[ColumnWeight] | None = None
    target_column: str | None = None
    threshold: float = 0.0
    random_state: int | None = None
    n_rows: int | None = None


class SplittingConfig(BaseModel):
    train: float
    validation: float

    @field_validator("train", "validation")
    @classmethod
    def check_ratio(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("Ratio must be between 0 and 1")
        return v


class PipelineConfig(BaseModel):
    version: str = "0.1"
    dataset_file: str
    dataset_url: str | None = None
    operations: list[OperationConfig]
    target: str
    splitting: SplittingConfig
    checkpoint_path: str | None = None
    store_format: Literal["csv"] = "csv"

    _config_filename: str = ""

    @cached_property
    def checkpoint_dir(self) -> Path | None:
        if self.checkpoint_path is None:
            return None
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = Path(self.checkpoint_path) / self._config_filename / self.version / timestamp
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_dataset(self, base_path: Path) -> None:
        """Download dataset if not present."""
        dataset_path = base_path / self.dataset_file
        if dataset_path.exists():
            return

        if self.dataset_url is None:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        logger.info("Downloading %s...", self.dataset_file)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(self.dataset_url, dataset_path)
        logger.info("Downloaded to %s", dataset_path)


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    config = PipelineConfig.model_validate(data)
    config._config_filename = path.stem
    return config
