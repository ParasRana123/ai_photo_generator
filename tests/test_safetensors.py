"""Unit tests for SafeTensors LoRA serialization and metadata inspection."""

import pytest
from src.training.model_loader import MockSDXLUNet
from src.training.lora_network import LoRANetworkManager
from src.training.saver import LoRASaver
from src.utils.config import TrainingConfig


def test_safetensors_save_and_verify(tmp_path):
    unet = MockSDXLUNet()
    manager = LoRANetworkManager(rank=8, alpha=16, target_modules=["cross_attn"])
    lora_unet = manager.inject_lora_unet(unet)

    saver = LoRASaver(
        output_dir=tmp_path,
        output_name="test_subject_lora",
        trigger_word="custom person"
    )
    cfg = TrainingConfig(rank=8, lora_alpha=16)
    saved_path = saver.save(lora_unet, training_config=cfg)

    assert saved_path.exists()
    assert saved_path.suffix == ".safetensors"

    # Inspect the saved file
    info = saver.inspect_safetensors(saved_path)
    assert info["tensor_count"] > 0
    assert info["file_size_bytes"] > 0


def test_key_name_standardization():
    saver = LoRASaver()
    raw_keys = {
        "base_model.model.down_blocks.0.attentions.0.to_q.lora_A.weight": None,
        "base_model.model.down_blocks.0.attentions.0.to_q.lora_B.weight": None,
    }
    converted = saver.convert_peft_to_diffusers_naming(raw_keys)
    for k in converted.keys():
        assert "lora_down.weight" in k or "lora_up.weight" in k
        assert k.startswith("lora_unet_")

