"""
SDXL Pipeline Loader and Optimizer.
Loads StableDiffusionXLPipeline, attaches custom VAE, configures advanced schedulers,
loads trained .safetensors LoRA adapters with weight scaling, and applies VRAM optimizations.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
import torch
from PIL import Image, ImageDraw

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoencoderKL,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    HeunDiscreteScheduler,
    LCMScheduler
)
from src.utils.inference_config import InferenceConfig
from src.utils.device import get_optimal_device, get_torch_dtype

logger = logging.getLogger("PipelineLoader")


class MockSDXLInferencePipeline:
    """
    Lightweight mock inference pipeline for offline verification, CI testing,
    and running without full 7GB SDXL downloads.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.active_lora_path: Optional[str] = None
        self.lora_scale: float = 1.0
        self.ip_adapter_scale: float = 0.7
        self.engine_mode: str = "sdxl_lora"
        self.fused: bool = False
        self.scheduler_name: str = "dpm++_2m_karras"
        self._has_active_lora: bool = True
        self._has_active_ip_adapter: bool = False

    def set_ip_adapter_scale(self, scale: float) -> None:
        self.ip_adapter_scale = scale

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Path], **kwargs) -> None:
        self.active_lora_path = str(pretrained_model_name_or_path_or_dict)
        logger.info(f"[MockPipeline] Loaded LoRA weights from: {self.active_lora_path}")

    def set_adapters(self, adapter_names: List[str], adapter_weights: List[float]) -> None:
        if adapter_weights:
            self.lora_scale = adapter_weights[0]

    def fuse_lora(self, lora_scale: float = 1.0) -> None:
        self.fused = True
        self.lora_scale = lora_scale

    def unfuse_lora(self) -> None:
        self.fused = False

    def enable_model_cpu_offload(self) -> None:
        pass

    def enable_vae_tiling(self) -> None:
        pass

    def enable_vae_slicing(self) -> None:
        pass

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Any] = None,
        callback_steps: int = 1,
        **kwargs
    ) -> Any:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        images = []

        seed = 42
        if generator is not None and hasattr(generator, "initial_seed"):
            seed = generator.initial_seed()

        for idx, p in enumerate(prompts):
            for img_idx in range(num_images_per_prompt):
                # Trigger callback steps if provided
                if callback:
                    for s in range(0, num_inference_steps, max(1, callback_steps)):
                        callback(s, num_inference_steps, None)

                # Generate clean synthetic render
                img = Image.new("RGB", (width, height), color=(25, 28, 36))
                draw = ImageDraw.Draw(img)

                # Portrait canvas composition
                margin = int(width * 0.05)
                draw.rectangle([margin, margin, width - margin, height - margin], outline=(80, 110, 150), width=3)
                draw.rectangle([margin + 20, margin + 20, width - margin - 20, int(height * 0.65)], fill=(45, 55, 75))

                # Stylized head & shoulders
                head_cx = width // 2
                head_cy = int(height * 0.35)
                head_rx = int(width * 0.18)
                head_ry = int(height * 0.22)
                draw.ellipse([head_cx - head_rx, head_cy - head_ry, head_cx + head_rx, head_cy + head_ry], fill=(225, 185, 155))
                draw.ellipse([head_cx - int(width * 0.35), int(height * 0.52), head_cx + int(width * 0.35), height], fill=(30, 35, 45))

                # Information plate
                draw.rectangle([margin + 20, int(height * 0.68), width - margin - 20, height - margin - 20], fill=(15, 18, 24))
                title = "[Phase 2: SDXL LoRA Inference]"
                draw.text((margin + 40, int(height * 0.70)), title, fill=(255, 255, 255))
                draw.text((margin + 40, int(height * 0.75)), f"Prompt: {p[:90]}...", fill=(200, 200, 210))
                draw.text((margin + 40, int(height * 0.81)), f"LoRA: {self.active_lora_path or 'None'} (Scale: {self.lora_scale})", fill=(120, 220, 140))
                draw.text((margin + 40, int(height * 0.86)), f"Sampler: {self.scheduler_name} | Steps: {num_inference_steps} | CFG: {guidance_scale} | Seed: {seed}", fill=(180, 180, 190))

                images.append(img)

        from types import SimpleNamespace
        return SimpleNamespace(images=images)


class SDXLPipelineLoader:
    """Manages loading, scheduler swapping, LoRA / IP-Adapter injection, and memory optimization."""

    _PIPELINE_CACHE: Dict[tuple, Any] = {}

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.device = (
            torch.device(self.config.device)
            if self.config.device
            else get_optimal_device()
        )
        # PyTorch CPU does not support fp16 inference for Conv2d/LayerNorm
        if self.device.type == "cpu":
            self.dtype = torch.float32
        else:
            self.dtype = get_torch_dtype(self.config.mixed_precision)

    def _configure_scheduler(self, pipeline: Any, sampler_name: str) -> None:
        """Configures the chosen sampling scheduler on the pipeline."""
        s = sampler_name.lower()
        base_config = pipeline.scheduler.config

        if s == "euler":
            pipeline.scheduler = EulerDiscreteScheduler.from_config(base_config)
        elif s == "euler_a":
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(base_config)
        elif s == "dpm++_2m":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                base_config, use_karras_sigmas=False
            )
        elif s == "dpm++_2m_karras":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                base_config, use_karras_sigmas=True
            )
        elif s == "ddim":
            pipeline.scheduler = DDIMScheduler.from_config(base_config)
        elif s == "heun":
            pipeline.scheduler = HeunDiscreteScheduler.from_config(base_config)
        elif s == "lcm":
            pipeline.scheduler = LCMScheduler.from_config(base_config)
        else:
            logger.warning(f"Unknown scheduler '{sampler_name}', retaining default scheduler.")

    def load_pipeline(
        self,
        mock: bool = False
    ) -> Any:
        """
        Loads either SDXL LoRA or Fast Instant FaceID + LCM pipeline, attaches scheduler,
        injects adapters/weights, and applies memory optimizations.
        """
        if mock:
            pipe = MockSDXLInferencePipeline(device=self.device)
            pipe.scheduler_name = self.config.sampler
            pipe.engine_mode = self.config.engine_mode
            pipe.ip_adapter_scale = self.config.ip_adapter_scale
            pipe._has_active_ip_adapter = (self.config.engine_mode == "instant_faceid")
            if Path(self.config.lora_weights_path).exists():
                pipe.load_lora_weights(self.config.lora_weights_path)
                pipe.lora_scale = self.config.lora_scale
            return pipe

        cache_key = (
            self.config.engine_mode,
            self.config.base_model,
            self.config.use_lcm,
            self.config.sampler,
            str(self.dtype),
            self.config.lora_weights_path if self.config.engine_mode == "sdxl_lora" else None,
        )
        if cache_key in self._PIPELINE_CACHE:
            cached_pipe = self._PIPELINE_CACHE[cache_key]
            if getattr(cached_pipe, "_has_active_ip_adapter", False) and hasattr(cached_pipe, "set_ip_adapter_scale"):
                cached_pipe.set_ip_adapter_scale(self.config.ip_adapter_scale)
            return cached_pipe

        try:
            logger.info(f"Loading diffusion pipeline ({self.config.engine_mode}) from {self.config.base_model}...")
            is_sdxl = "xl" in self.config.base_model.lower()
            use_fp16_variant = (self.dtype == torch.float16)

            if is_sdxl:
                vae = None
                if self.config.vae_model:
                    try:
                        vae = AutoencoderKL.from_pretrained(
                            self.config.vae_model,
                            torch_dtype=self.dtype
                        )
                    except Exception as e:
                        logger.warning(f"Could not load custom VAE ({e}), using default VAE.")

                pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.config.base_model,
                    vae=vae,
                    torch_dtype=self.dtype,
                    variant="fp16" if use_fp16_variant else None
                )
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    self.config.base_model,
                    torch_dtype=self.dtype
                )

            # Disable safety checker for local artistic generation if present
            if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
                pipe.safety_checker = None

            pipe._has_active_lora = False
            pipe._has_active_ip_adapter = False

            # Fast LCM-LoRA acceleration (fuses cleanly before IP-Adapter FaceID)
            if self.config.use_lcm or self.config.sampler.lower() == "lcm":
                try:
                    logger.info(f"Loading & fusing LCM-LoRA ({self.config.lcm_lora_repo})...")
                    pipe.load_lora_weights(self.config.lcm_lora_repo)
                    if hasattr(pipe, "fuse_lora"):
                        pipe.fuse_lora()
                    if hasattr(pipe, "unload_lora_weights"):
                        pipe.unload_lora_weights()
                    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                except Exception as lcm_err:
                    logger.warning(f"Could not apply LCM-LoRA ({lcm_err}); using standard scheduler.")
                    self._configure_scheduler(pipe, self.config.sampler)
            else:
                self._configure_scheduler(pipe, self.config.sampler)

            # Attach IP-Adapter-FaceID if running in Instant FaceID mode
            if self.config.engine_mode == "instant_faceid":
                logger.info(
                    f"Loading IP-Adapter-FaceID ({self.config.ip_adapter_repo}/{self.config.ip_adapter_weight_name})..."
                )
                pipe.load_ip_adapter(
                    self.config.ip_adapter_repo,
                    subfolder=None,
                    weight_name=self.config.ip_adapter_weight_name,
                    image_encoder_folder=None
                )
                pipe.set_ip_adapter_scale(self.config.ip_adapter_scale)
                pipe._has_active_ip_adapter = True

            # Attach SDXL trained LoRA weights if running in SDXL LoRA mode
            lora_path = Path(self.config.lora_weights_path)
            if self.config.engine_mode == "sdxl_lora" and is_sdxl and lora_path.exists():
                from safetensors.torch import load_file
                lora_tensors = load_file(str(lora_path))
                is_mock_lora = any("cross_attn" in k for k in lora_tensors.keys()) and not any(
                    "down_blocks" in k or "up_blocks" in k for k in lora_tensors.keys()
                )
                if is_mock_lora:
                    logger.warning(
                        f"Skipping '{lora_path.name}' because it was saved during a Dry-Run mock session. "
                        "Train on a GPU without --dry-run to apply real face LoRA weights."
                    )
                else:
                    logger.info(f"Loading trained LoRA from: {lora_path} with scale {self.config.lora_scale}")
                    pipe.load_lora_weights(str(lora_path))
                    pipe._has_active_lora = True

                    if self.config.fuse_lora and hasattr(pipe, "fuse_lora"):
                        pipe.fuse_lora(lora_scale=self.config.lora_scale)
                    elif hasattr(pipe, "set_adapters"):
                        pipe.set_adapters(["default"], adapter_weights=[self.config.lora_scale])

            # Apply memory optimizations
            if self.device.type == "cuda":
                if self.config.enable_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(self.device)

                if self.config.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()
                if self.config.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
            else:
                pipe.to(self.device)
                if self.config.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()

            self._PIPELINE_CACHE[cache_key] = pipe
            return pipe

        except Exception as err:
            raise RuntimeError(
                f"Failed to load diffusion pipeline ({self.config.base_model}) on {self.device.type.upper()}: {err}"
            ) from err

