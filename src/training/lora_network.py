"""
LoRA network injection and parameter management for SDXL.
Injects low-rank adapter matrices into UNet cross-attention layers using PEFT / Diffusers,
freezes non-adapter weights, and provides parameter statistics.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class LoRANetworkManager:
    """Manages injection of LoRA adapters into SDXL models and tracks trainable weights."""

    # Default SDXL UNet cross-attention targets for facial fine-tuning
    DEFAULT_UNET_TARGET_MODULES = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]

    def __init__(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or self.DEFAULT_UNET_TARGET_MODULES

    def create_lora_config(self) -> LoraConfig:
        """Creates a PEFT LoraConfig targeted for diffusion attention layers."""
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            init_lora_weights="gaussian",
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias="none"
        )

    def inject_lora_unet(self, unet: nn.Module) -> nn.Module:
        """
        Injects trainable LoRA matrices into the UNet model.
        Freezes all existing weights and unfreezes only newly injected LoRA parameters.
        """
        # Ensure base weights are frozen
        unet.requires_grad_(False)

        # Apply PEFT LoRA
        lora_config = self.create_lora_config()
        lora_unet = get_peft_model(unet, lora_config)

        return lora_unet

    @staticmethod
    def get_parameter_summary(model: nn.Module) -> Dict[str, Any]:
        """Calculates total, frozen, and trainable parameter counts."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        percentage = (trainable_params / total_params * 100) if total_params > 0 else 0.0

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_percentage": round(percentage, 3)
        }

    @staticmethod
    def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Extracts only the trainable LoRA adapter tensors from the model state dict.
        Filters out frozen base model weights.
        """
        state_dict = model.state_dict()
        lora_dict = {}

        for k, v in state_dict.items():
            if "lora" in k.lower() or "adapter" in k.lower():
                lora_dict[k] = v.detach().cpu()

        # If PEFT prefixes exist, make sure keys are cleanly identified
        if not lora_dict:
            # Fallback check for requires_grad tensors
            for name, param in model.named_parameters():
                if param.requires_grad:
                    lora_dict[name] = param.detach().cpu()

        return lora_dict

