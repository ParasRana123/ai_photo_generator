"""Unit tests for configuration schema and validation."""

import pytest
from pathlib import Path
from src.utils.config import (
    PreprocessConfig,
    ModelConfig,
    TrainingConfig,
    ValidationConfig,
    PipelineConfig
)


def test_preprocess_config_defaults():
    cfg = PreprocessConfig()
    cfg.validate()
    assert cfg.resolution == 1024
    assert cfg.trigger_word == "sks person"
    assert cfg.min_photos == 10
    assert cfg.max_photos == 30


def test_preprocess_config_invalid_resolution():
    cfg = PreprocessConfig(resolution=1025)
    with pytest.raises(ValueError, match="divisible by 8"):
        cfg.validate()


def test_training_config_validation():
    cfg = TrainingConfig()
    cfg.validate()
    assert cfg.rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.learning_rate == 1e-4

    invalid_rank = TrainingConfig(rank=0)
    with pytest.raises(ValueError, match="rank must be positive"):
        invalid_rank.validate()

    invalid_lr = TrainingConfig(learning_rate=-0.01)
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        invalid_lr.validate()


def test_pipeline_config_yaml_roundtrip(tmp_path):
    yaml_file = tmp_path / "test_config.yaml"
    cfg = PipelineConfig()
    cfg.training.rank = 32
    cfg.preprocess.trigger_word = "test subject"
    cfg.to_yaml(yaml_file)

    loaded = PipelineConfig.from_yaml(yaml_file)
    assert loaded.training.rank == 32
    assert loaded.preprocess.trigger_word == "test subject"
    assert loaded.preprocess.resolution == 1024

