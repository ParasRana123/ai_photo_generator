"""Unit tests for LoRA adapter injection and parameter tracking."""

import torch
from src.training.model_loader import MockSDXLUNet
from src.training.lora_network import LoRANetworkManager


def test_lora_network_injection():
    unet = MockSDXLUNet()
    manager = LoRANetworkManager(rank=16, alpha=32, target_modules=["cross_attn"])
    lora_unet = manager.inject_lora_unet(unet)

    summary = manager.get_parameter_summary(lora_unet)
    assert summary["trainable_params"] > 0
    assert summary["frozen_params"] > 0
    assert summary["trainable_percentage"] < 10.0  # LoRA should be a small fraction


def test_lora_state_dict_extraction():
    unet = MockSDXLUNet()
    manager = LoRANetworkManager(rank=8, alpha=16, target_modules=["cross_attn"])
    lora_unet = manager.inject_lora_unet(unet)

    lora_state = manager.extract_lora_state_dict(lora_unet)
    assert len(lora_state) > 0
    for name, tensor in lora_state.items():
        assert isinstance(tensor, torch.Tensor)

