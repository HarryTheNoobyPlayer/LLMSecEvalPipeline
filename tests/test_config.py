"""Unit tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llmseceval.config import (
    BanditConfig,
    DataLoaderConfig,
    PipelineConfig,
    SASTExecutorConfig,
    load_config,
)


class TestDataLoaderConfig:
    def test_defaults(self) -> None:
        cfg = DataLoaderConfig()
        assert cfg.min_char_length == 10
        assert cfg.language == "en"
        assert cfg.deduplicate is True
        assert cfg.sample_size is None
        assert cfg.random_seed == 42

    def test_valid_language(self) -> None:
        cfg = DataLoaderConfig(language="FR")
        assert cfg.language == "fr"  # normalised to lowercase

    def test_invalid_language_too_long(self) -> None:
        with pytest.raises(ValidationError, match="2-letter"):
            DataLoaderConfig(language="eng")

    def test_invalid_language_numeric(self) -> None:
        with pytest.raises(ValidationError, match="2-letter"):
            DataLoaderConfig(language="12")

    def test_min_char_length_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DataLoaderConfig(min_char_length=0)

    def test_sample_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DataLoaderConfig(sample_size=0)

    def test_sample_size_none_is_valid(self) -> None:
        cfg = DataLoaderConfig(sample_size=None)
        assert cfg.sample_size is None


class TestSASTExecutorConfig:
    def test_defaults(self) -> None:
        cfg = SASTExecutorConfig()
        assert cfg.tool == "bandit"
        assert isinstance(cfg.bandit, BanditConfig)
        assert cfg.bandit.severity_threshold == "LOW"
        assert cfg.bandit.confidence_threshold == "LOW"
        assert cfg.bandit.extra_args == []
        assert cfg.timeout_per_file_s == 30
        assert cfg.parallel_workers == 4

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SASTExecutorConfig(timeout_per_file_s=0)

    def test_parallel_workers_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SASTExecutorConfig(parallel_workers=0)

    def test_accepts_extra_args(self) -> None:
        cfg = SASTExecutorConfig(bandit=BanditConfig(extra_args=["--skip", "B101"]))
        assert cfg.bandit.extra_args == ["--skip", "B101"]


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.log_level == "INFO"
        assert len(cfg.stages) == 4

    def test_invalid_log_level(self) -> None:
        with pytest.raises(ValidationError, match="log_level"):
            PipelineConfig(log_level="VERBOSE")

    def test_invalid_stage(self) -> None:
        with pytest.raises(ValidationError, match="Unknown stage"):
            PipelineConfig(stages=["data_loader", "nonexistent"])


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        config_data = {
            "pipeline": {
                "name": "test-run",
                "log_level": "DEBUG",
                "random_seed": 123,
                "stages": ["data_loader"],
            },
            "data_loader": {
                "dataset_path": "./test_data/",
                "language": "en",
                "min_char_length": 20,
            },
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as fh:
            yaml.dump(config_data, fh)

        cfg = load_config(config_file)
        assert cfg.name == "test-run"
        assert cfg.log_level == "DEBUG"
        assert cfg.data_loader.min_char_length == 20

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        cfg = load_config(config_file)
        assert cfg.name == "deepseek-bandit-devgpt"
        assert cfg.data_loader.min_char_length == 10

    def test_loads_real_config(self) -> None:
        """Validate the actual config.yaml in the project root."""
        real_config = Path(__file__).parent.parent / "config.yaml"
        if real_config.exists():
            cfg = load_config(real_config)
            assert cfg.name == "deepseek-bandit-devgpt"
            assert cfg.data_loader.language == "en"
