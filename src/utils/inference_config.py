"""
Configuration schema and validation for Phase 2: SDXL LoRA Inference Pipeline.
Manages base model loading, LoRA weight scaling, sampling schedulers,
prompt parameters, and memory optimization flags.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import yaml
import json


VALID_SAMPLERS = [
    "euler",
    "euler_a",
    "dpm++_2m",
    "dpm++_2m_karras",
    "ddim",
    "heun",
    "lms",
    "lcm"
]

VALID_ENGINE_MODES = ["sdxl_lora", "instant_faceid", "cloud_zerogpu", "google_imagen3", "fal_ai", "openai_dalle3"]
VALID_CLOUD_PROVIDERS = ["auto", "pulid_flux", "instantid", "photomaker_v2", "pulid_sdxl"]


@dataclass
class InferenceConfig:
    """Configuration for running inference with SDXL LoRA, Fast FaceID, Free Cloud ZeroGPU, or Direct APIs."""

    # Engine Mode & Model Weights
    engine_mode: str = "sdxl_lora"  # "sdxl_lora", "instant_faceid", "cloud_zerogpu", "google_imagen3", "fal_ai", "openai_dalle3"
    api_key: Optional[str] = None
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_model: Optional[str] = "madebyollin/sdxl-vae-fp16-fix"
    lora_weights_path: str = "checkpoints/person_sdxl_lora.safetensors"
    lora_scale: float = 0.85
    fuse_lora: bool = False

    # Fast Instant FaceID + LCM Settings
    face_id_path: str = "checkpoints/person_face_id.safetensors"
    ip_adapter_repo: str = "h94/IP-Adapter-FaceID"
    ip_adapter_weight_name: str = "ip-adapter-faceid_sd15.bin"
    ip_adapter_scale: float = 0.7
    use_lcm: bool = False
    lcm_lora_repo: str = "latent-consistency/lcm-lora-sdv1-5"

    # Free Cloud ZeroGPU (fal.ai Free Alternative) Settings
    cloud_provider: str = "auto"  # "auto", "pulid_flux", "instantid", "pulid_sdxl"
    cloud_profile_path: str = "checkpoints/cloud_face_profile.json"
    reference_image_path: Optional[str] = None
    id_weight: float = 1.0
    hf_token: Optional[str] = None

    # Prompt & Identity
    prompt: str = "A professional studio portrait of sks person in a tailored navy suit, sharp focus, 8k, cinematic lighting"
    negative_prompt: str = (
        "blurry, bad quality, low resolution, distorted face, bad eyes, extra limbs, ugly, poorly drawn, oversaturated"
    )
    style_preset: Optional[str] = "photorealistic"
    trigger_word: str = "sks person"
    auto_inject_trigger: bool = True

    # Generation parameters
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    seed: Optional[int] = 42
    num_images_per_prompt: int = 1
    sampler: str = "dpm++_2m_karras"

    # Memory and compute settings
    mixed_precision: str = "fp16"  # "fp16", "bf16", "fp32"
    device: Optional[str] = None  # None for auto-detect (cuda -> mps -> cpu)
    enable_cpu_offload: bool = True
    enable_vae_tiling: bool = True
    enable_vae_slicing: bool = True

    # Output settings
    output_dir: str = "outputs/generated"
    save_format: str = "PNG"
    embed_metadata: bool = True

    def validate(self) -> None:
        """Validates inference parameters."""
        if self.engine_mode not in VALID_ENGINE_MODES:
            raise ValueError(f"Invalid engine_mode '{self.engine_mode}'. Choose from {VALID_ENGINE_MODES}")
        if self.cloud_provider not in VALID_CLOUD_PROVIDERS:
            raise ValueError(f"Invalid cloud_provider '{self.cloud_provider}'. Choose from {VALID_CLOUD_PROVIDERS}")
        if self.id_weight < 0.0 or self.id_weight > 2.0:
            raise ValueError(f"id_weight should be between 0.0 and 2.0, got {self.id_weight}")
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise ValueError(f"Width ({self.width}) and Height ({self.height}) must be divisible by 8.")
        if self.width < 512 or self.height < 512:
            raise ValueError(f"Dimensions ({self.width}x{self.height}) must be at least 512x512.")
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {self.num_inference_steps}")
        if self.guidance_scale < 1.0:
            raise ValueError(f"guidance_scale must be >= 1.0, got {self.guidance_scale}")
        if self.lora_scale < 0.0 or self.lora_scale > 2.0:
            raise ValueError(f"lora_scale should be between 0.0 and 2.0, got {self.lora_scale}")
        if self.ip_adapter_scale < 0.0 or self.ip_adapter_scale > 2.0:
            raise ValueError(f"ip_adapter_scale should be between 0.0 and 2.0, got {self.ip_adapter_scale}")
        if self.sampler.lower() not in VALID_SAMPLERS:
            raise ValueError(f"Unknown sampler: {self.sampler}. Choose from {VALID_SAMPLERS}")
        if self.mixed_precision not in ("fp16", "bf16", "fp32", "no"):
            raise ValueError(f"Invalid precision: {self.mixed_precision}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "InferenceConfig":
        """Load inference configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Allow nested 'inference' key or top-level keys
        payload = data.get("inference", data)
        cfg = cls(**payload)
        cfg.validate()
        return cfg

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Export configuration to a YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"inference": asdict(self)}, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
