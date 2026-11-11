"""
Core SDXL LoRA Inference and Image Generation Engine.
Orchestrates prompt building, pipeline execution, deterministic seed control,
progress callbacks, and image persistence with metadata embedding.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import time
import logging
import torch
from PIL import Image

from src.utils.inference_config import InferenceConfig
from src.utils.device import get_optimal_device, clear_memory
from src.training.face_encoder import FaceIdentityEncoder
from .pipeline_loader import SDXLPipelineLoader
from .prompt_enhancer import PromptEnhancer
from .metadata_writer import PNGMetadataWriter

logger = logging.getLogger("InferenceEngine")


@dataclass
class GenerationResult:
    """Encapsulates generated images, execution metadata, and file paths."""
    images: List[Image.Image]
    prompt: str
    negative_prompt: str
    seed: int
    saved_paths: List[str]
    metadata: Dict[str, Any]
    duration_seconds: float


class SDXLInferenceEngine:
    """High-level generator interface for SDXL LoRA personal photo synthesis."""

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        mock: bool = False
    ):
        self.config = config or InferenceConfig()
        self.mock = mock
        self.device = (
            torch.device(self.config.device)
            if self.config.device
            else get_optimal_device()
        )

        auto_inject = (
            self.config.auto_inject_trigger
            if self.config.engine_mode != "instant_faceid"
            else False
        )
        self.enhancer = PromptEnhancer(
            trigger_word=self.config.trigger_word,
            auto_inject_trigger=auto_inject
        )

        self.loader = SDXLPipelineLoader(self.config)
        self.pipeline = self.loader.load_pipeline(mock=self.mock)

    def generate(
        self,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        style_preset: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        num_images: int = 1,
        callback: Optional[Any] = None,
        save: bool = True
    ) -> GenerationResult:
        """
        Executes personal image generation with the configured SDXL + LoRA pipeline.
        """
        start_time = time.time()

        # Resolve parameters
        raw_prompt = prompt or self.config.prompt
        raw_neg = negative_prompt or self.config.negative_prompt
        selected_style = style_preset if style_preset is not None else self.config.style_preset
        w = width or self.config.width
        h = height or self.config.height
        steps = num_inference_steps or self.config.num_inference_steps
        cfg_scale = guidance_scale or self.config.guidance_scale
        current_seed = seed if seed is not None else (self.config.seed or int(time.time()) % 100000)

        # 1. Enhance and assemble prompts
        pos_prompt, neg_prompt = self.enhancer.build_prompts(
            raw_prompt,
            negative_prompt=raw_neg,
            style_preset_name=selected_style
        )

        # 2. Configure deterministic generator
        generator = torch.Generator(device="cpu").manual_seed(current_seed)

        # 3. Prepare LoRA cross-attention scale and/or FaceID IP-Adapter embeddings
        call_kwargs: Dict[str, Any] = {
            "prompt": pos_prompt,
            "negative_prompt": neg_prompt,
            "width": w,
            "height": h,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "num_images_per_prompt": num_images,
            "generator": generator,
        }

        if getattr(self.pipeline, "_has_active_ip_adapter", False):
            face_path = Path(self.config.face_id_path)
            if face_path.exists():
                face_embed = FaceIdentityEncoder.load_face_embedding(face_path)
            elif self.mock:
                face_embed = torch.randn(1, 1, 512, dtype=torch.float32)
            else:
                raise FileNotFoundError(
                    f"FaceID embedding not found at '{face_path}'. Run Instant FaceID enrollment first."
                )

            do_cfg = cfg_scale > 1.0
            face_embed = face_embed.to(dtype=self.loader.dtype)
            call_kwargs["ip_adapter_image_embeds"] = FaceIdentityEncoder.prepare_ip_adapter_embeds(
                face_embed, do_classifier_free_guidance=do_cfg
            )
        elif self.mock or getattr(self.pipeline, "_has_active_lora", False):
            call_kwargs["cross_attention_kwargs"] = {"scale": self.config.lora_scale}

        if callback is not None:
            call_kwargs["callback"] = callback

        logger.info(f"Generating {num_images} image(s) [{self.config.engine_mode}] with seed {current_seed}...")
        logger.info(f"  Prompt: {pos_prompt[:100]}...")

        # 4. Execute diffusion inference
        pipe_output = self.pipeline(**call_kwargs)

        images = pipe_output.images if hasattr(pipe_output, "images") else pipe_output
        duration = time.time() - start_time

        # 5. Build metadata
        meta = {
            "engine_mode": self.config.engine_mode,
            "prompt": pos_prompt,
            "raw_prompt": raw_prompt,
            "negative_prompt": neg_prompt,
            "style_preset": selected_style,
            "trigger_word": self.config.trigger_word,
            "width": w,
            "height": h,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "seed": current_seed,
            "sampler": self.config.sampler,
            "use_lcm": self.config.use_lcm,
            "base_model": self.config.base_model,
            "lora_name": Path(self.config.lora_weights_path).name,
            "lora_scale": self.config.lora_scale,
            "face_id_name": Path(self.config.face_id_path).name,
            "ip_adapter_scale": self.config.ip_adapter_scale,
            "generation_time_seconds": round(duration, 3)
        }

        # 6. Save results with embedded metadata
        saved_paths: List[str] = []
        if save:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            for i, img in enumerate(images):
                filename = f"gen_{timestamp}_{current_seed}_{i:02d}.png"
                out_path = out_dir / filename
                PNGMetadataWriter.save_image_with_metadata(img, out_path, meta)
                saved_paths.append(str(out_path))
                logger.info(f"  Saved image: {out_path}")

        clear_memory()

        return GenerationResult(
            images=images,
            prompt=pos_prompt,
            negative_prompt=neg_prompt,
            seed=current_seed,
            saved_paths=saved_paths,
            metadata=meta,
            duration_seconds=round(duration, 3)
        )

    def generate_batch(
        self,
        prompts: List[str],
        style_preset: Optional[str] = None,
        base_seed: int = 42
    ) -> List[GenerationResult]:
        """Runs batch generation across a sequence of prompts."""
        results = []
        for idx, p in enumerate(prompts):
            res = self.generate(
                prompt=p,
                style_preset=style_preset,
                seed=base_seed + idx
            )
            results.append(res)
        return results

