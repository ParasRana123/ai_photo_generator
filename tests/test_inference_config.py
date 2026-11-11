"""Unit tests for Phase 2 inference configuration schema and validation."""

import pytest
from src.utils.inference_config import InferenceConfig, VALID_SAMPLERS


def test_inference_config_defaults():
    cfg = InferenceConfig()
    cfg.validate()
    assert cfg.width == 1024
    assert cfg.height == 1024
    assert cfg.num_inference_steps == 30
    assert cfg.guidance_scale == 7.0
    assert cfg.lora_scale == 0.85
    assert cfg.sampler == "dpm++_2m_karras"


def test_inference_config_dimension_validation():
    # Width not divisible by 8
    cfg_bad_div = InferenceConfig(width=1025)
    with pytest.raises(ValueError, match="divisible by 8"):
        cfg_bad_div.validate()

    # Dimension too small
    cfg_small = InferenceConfig(width=256, height=256)
    with pytest.raises(ValueError, match="at least 512x512"):
        cfg_small.validate()


def test_inference_config_invalid_sampler():
    cfg = InferenceConfig(sampler="non_existent_sampler")
    with pytest.raises(ValueError, match="Unknown sampler"):
        cfg.validate()


def test_inference_config_yaml_roundtrip(tmp_path):
    yaml_file = tmp_path / "inference_test.yaml"
    cfg = InferenceConfig(
        prompt="A photo of sks person on Mars",
        style_preset="cyberpunk",
        lora_scale=0.9,
        num_inference_steps=40
    )
    cfg.to_yaml(yaml_file)

    loaded = InferenceConfig.from_yaml(yaml_file)
    assert loaded.prompt == "A photo of sks person on Mars"
    assert loaded.style_preset == "cyberpunk"
    assert loaded.lora_scale == 0.9
    assert loaded.num_inference_steps == 40

