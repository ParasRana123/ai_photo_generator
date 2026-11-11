"""
SDXL model loading utility.
Loads UNet2DConditionModel, VAE, dual text encoders, and noise schedulers
with precision handling and parameter freezing.
"""

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from src.utils.config import ModelConfig
from src.utils.device import get_torch_dtype


class ModelLoader:
    """Manages loading, precision casting, and freezing of SDXL components."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.dtype = get_torch_dtype(self.config.variant or "fp32")

    def load_tokenizers(self) -> Tuple[CLIPTokenizer, CLIPTokenizer]:
        """Loads both CLIP tokenizers required for SDXL prompt encoding."""
        pretrained = self.config.pretrained_model_name_or_path
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            pretrained,
            subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained,
            subfolder="tokenizer_2"
        )
        return tokenizer_1, tokenizer_2

    def load_vae(self, device: torch.device) -> AutoencoderKL:
        """Loads the VAE (optionally using fp16 fix to avoid NaN latents)."""
        vae_path = self.config.vae_name_or_path or self.config.pretrained_model_name_or_path
        subfolder = None if self.config.vae_name_or_path else "vae"

        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder=subfolder,
            torch_dtype=self.dtype
        )
        vae.requires_grad_(False)
        vae.to(device)
        return vae

    def load_text_encoders(
        self,
        device: torch.device
    ) -> Tuple[CLIPTextModel, CLIPTextModelWithProjection]:
        """Loads both text encoders in frozen mode."""
        pretrained = self.config.pretrained_model_name_or_path

        text_encoder_1 = CLIPTextModel.from_pretrained(
            pretrained,
            subfolder="text_encoder",
            torch_dtype=self.dtype
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained,
            subfolder="text_encoder_2",
            torch_dtype=self.dtype
        )

        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        text_encoder_1.to(device)
        text_encoder_2.to(device)

        return text_encoder_1, text_encoder_2

    def load_unet(self, device: torch.device) -> UNet2DConditionModel:
        """Loads the SDXL UNet backbone with frozen base weights."""
        pretrained = self.config.pretrained_model_name_or_path
        unet = UNet2DConditionModel.from_pretrained(
            pretrained,
            subfolder="unet",
            torch_dtype=self.dtype
        )
        unet.requires_grad_(False)

        if self.config.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()

        unet.to(device)
        return unet

    def load_noise_scheduler(self) -> DDPMScheduler:
        """Loads the training DDPM noise scheduler."""
        pretrained = self.config.pretrained_model_name_or_path
        scheduler = DDPMScheduler.from_pretrained(
            pretrained,
            subfolder="scheduler"
        )
        return scheduler


class MockSDXLUNet(nn.Module):
    """
    Lightweight mock UNet architecture matching SDXL input/output tensors.
    Used for rapid local tests, continuous integration, and dry-run execution.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(320, 64)
        self.text_proj = nn.Linear(2048, 64)
        self.cross_attn = nn.Linear(64, 64)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True
    ) -> Any:
        h = self.conv_in(sample)
        # Apply dummy cross-attention projection
        attn_out = self.cross_attn(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.conv_out(h + attn_out)

        if return_dict:
            from types import SimpleNamespace
            return SimpleNamespace(sample=out)
        return (out,)

