"""
Configuration management system for the Phase 1 AI Photo Generator pipeline.
Handles configuration schemas for preprocessing, model loading, LoRA training,
and validation with YAML/JSON serialization.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import yaml
import json


@dataclass
class PreprocessConfig:
    """Configuration for photo ingestion, face framing, and resizing."""
    input_dir: str = "data/raw"
    output_dir: str = "data/processed"
    resolution: int = 1024
    min_photos: int = 10
    max_photos: int = 30
    trigger_word: str = "sks person"
    face_padding: float = 0.35
    detect_face: bool = True
    aspect_ratio_buckets: List[List[int]] = field(
        default_factory=lambda: [
            [1024, 1024],
            [896, 1152],
            [1152, 896],
            [768, 1344],
            [1344, 768]
        ]
    )
    supported_formats: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"]
    )

    def validate(self) -> None:
        if self.resolution % 8 != 0:
            raise ValueError(f"Resolution must be divisible by 8, got {self.resolution}")
        if self.min_photos < 1:
            raise ValueError("min_photos must be >= 1")
        if self.max_photos < self.min_photos:
            raise ValueError("max_photos must be >= min_photos")


@dataclass
class ModelConfig:
    """Configuration for base SDXL models and tokenizers."""
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_name_or_path: Optional[str] = "madebyollin/sdxl-vae-fp16-fix"
    revision: Optional[str] = None
    variant: Optional[str] = "fp16"
    enable_xformers: bool = False
    gradient_checkpointing: bool = True


@dataclass
class TrainingConfig:
    """Hyperparameters and settings for SDXL LoRA fine-tuning."""
    output_dir: str = "checkpoints"
    output_name: str = "person_sdxl_lora"
    rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    train_unet: bool = True
    train_text_encoder: bool = False
    learning_rate: float = 1e-4
    text_encoder_lr: float = 5e-5
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 50
    max_train_steps: int = 500
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    seed: int = 42
    cache_latents: bool = True
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    checkpointing_steps: int = 250
    save_safetensors: bool = True

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        if self.mixed_precision not in ("no", "fp16", "bf16"):
            raise ValueError(f"Invalid mixed precision: {self.mixed_precision}")


@dataclass
class ValidationConfig:
    """Configuration for testing and evaluating the trained LoRA."""
    validation_prompts: List[str] = field(
        default_factory=lambda: [
            "A professional studio portrait photo of sks person in a tailored dark suit, sharp focus, 8k, cinematic lighting",
            "A casual candid photo of sks person sitting at a coffee shop on a sunny morning, natural lighting",
            "A close-up headshot of sks person with soft rim lighting, highly detailed eyes, 85mm lens photograph",
            "An epic cinematic portrait of sks person in a cyberpunk neon lit city at night, rain reflections"
        ]
    )
    negative_prompt: str = (
        "blurry, bad quality, low resolution, distorted face, extra limbs, ugly, poorly drawn, oversaturated"
    )
    num_validation_images: int = 4
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = 42
    output_dir: str = "outputs/validation"


@dataclass
class PipelineConfig:
    """Master configuration encompassing the whole Phase 1 pipeline."""
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def validate(self) -> None:
        self.preprocess.validate()
        self.training.validate()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        preprocess = PreprocessConfig(**data.get("preprocess", {}))
        model = ModelConfig(**data.get("model", {}))
        training = TrainingConfig(**data.get("training", {}))
        validation = ValidationConfig(**data.get("validation", {}))

        cfg = cls(
            preprocess=preprocess,
            model=model,
            training=training,
            validation=validation
        )
        cfg.validate()
        return cfg

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Export configuration to a YAML file."""
        data = {
            "preprocess": asdict(self.preprocess),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "validation": asdict(self.validation)
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preprocess": asdict(self.preprocess),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "validation": asdict(self.validation)
        }
