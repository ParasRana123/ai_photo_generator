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
from src.training.reference_selector import ReferencePhotoSelector
from .pipeline_loader import SDXLPipelineLoader
from .cloud_client import CloudZeroGPUClient
from .api_providers import DirectAPIProvider
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
            if self.config.engine_mode not in ("instant_faceid", "cloud_zerogpu", "google_imagen3", "fal_ai", "openai_dalle3")
            else False
        )
        self.enhancer = PromptEnhancer(
            trigger_word=self.config.trigger_word,
            auto_inject_trigger=auto_inject
        )

        if self.config.engine_mode in ("cloud_zerogpu", "google_imagen3", "fal_ai", "openai_dalle3"):
            self.loader = None
            self.pipeline = None
            self.cloud_client = (
                CloudZeroGPUClient(hf_token=self.config.hf_token)
                if self.config.engine_mode == "cloud_zerogpu"
                else None
            )
        else:
            self.cloud_client = None
            self.loader = SDXLPipelineLoader(self.config)
            self.pipeline = self.loader.load_pipeline(mock=self.mock)

    def _resolve_cloud_reference_photos(self) -> List[str]:
        """Resolves top ranked reference photos from profile or data/raw for cloud synthesis."""
        if self.config.reference_image_path and Path(self.config.reference_image_path).exists():
            return [self.config.reference_image_path]

        profile = ReferencePhotoSelector.load_profile(self.config.cloud_profile_path)
        if profile and profile.get("top_photos"):
            valid = [p for p in profile["top_photos"] if Path(p).exists()]
            if valid:
                return valid

        raw_dir = Path("data/raw")
        if raw_dir.exists():
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            imgs = sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
            user_imgs = [str(p) for p in imgs if not p.name.startswith("sample_photo_")]
            if user_imgs:
                return user_imgs[:4]
            if imgs:
                return [str(p) for p in imgs[:4]]

        if self.mock:
            return ["mock_reference_face.png"]

        raise FileNotFoundError(
            "No reference photos found in 'data/raw/'. Upload 10-15 photos of yourself first!"
        )

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

        # Cloud ZeroGPU Path (fal.ai free alternative)
        if self.config.engine_mode == "cloud_zerogpu":
            ref_photos = self._resolve_cloud_reference_photos()
            images: List[Image.Image] = []
            cloud_meta: Dict[str, Any] = {}
            for i in range(num_images):
                img, cm = self.cloud_client.generate(
                    reference_photos=ref_photos,
                    prompt=pos_prompt,
                    negative_prompt=neg_prompt,
                    provider=self.config.cloud_provider,
                    width=w,
                    height=h,
                    steps=steps,
                    guidance_scale=cfg_scale,
                    id_weight=self.config.id_weight,
                    seed=current_seed + i,
                    mock=self.mock,
                )
                images.append(img)
                cloud_meta.update(cm)

            duration = time.time() - start_time
            meta = {
                "engine_mode": self.config.engine_mode,
                "cloud_provider": cloud_meta.get("provider_used", self.config.cloud_provider),
                "cloud_space": cloud_meta.get("cloud_space", ""),
                "base_model": cloud_meta.get("cloud_model", self.config.base_model),
                "prompt": pos_prompt,
                "raw_prompt": raw_prompt,
                "negative_prompt": neg_prompt,
                "style_preset": selected_style,
                "width": images[0].size[0],
                "height": images[0].size[1],
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "id_weight": self.config.id_weight,
                "seed": current_seed,
                "reference_photo": Path(ref_photos[0]).name,
                "generation_time_seconds": round(duration, 3)
            }
        elif self.config.engine_mode == "google_imagen3":
            images = []
            api_meta = {}
            if self.mock:
                images.append(Image.new("RGB", (1024, 1024), color=(30, 40, 60)))
                api_meta = {"provider_name": "Google Gemini (Imagen 3 Mock)"}
            else:
                for _ in range(num_images):
                    img, meta_data = DirectAPIProvider.generate_gemini_imagen(
                        prompt=pos_prompt,
                        api_key=self.config.api_key,
                        num_images=1
                    )
                    images.append(img)
                    api_meta.update(meta_data)
            duration = time.time() - start_time
            meta = {
                "engine_mode": self.config.engine_mode,
                "api_provider": "google_imagen3",
                "base_model": "Google Imagen 3 (Gemini)",
                "prompt": pos_prompt,
                "raw_prompt": raw_prompt,
                "negative_prompt": neg_prompt,
                "style_preset": selected_style,
                "width": images[0].size[0],
                "height": images[0].size[1],
                "seed": current_seed,
                "generation_time_seconds": round(duration, 3)
            }
        elif self.config.engine_mode == "fal_ai":
            ref_photos = self._resolve_cloud_reference_photos()
            images = []
            api_meta = {}
            if self.mock:
                images.append(Image.new("RGB", (1024, 1024), color=(25, 30, 45)))
                api_meta = {"provider_name": "fal.ai (Mock)"}
            else:
                for _ in range(num_images):
                    img, meta_data = DirectAPIProvider.generate_fal_ai(
                        prompt=pos_prompt,
                        reference_photo=ref_photos[0],
                        api_key=self.config.api_key,
                        steps=steps,
                        guidance_scale=cfg_scale
                    )
                    images.append(img)
                    api_meta.update(meta_data)
            duration = time.time() - start_time
            meta = {
                "engine_mode": self.config.engine_mode,
                "api_provider": "fal_ai",
                "base_model": "fal.ai FLUX.1-dev PuLID",
                "prompt": pos_prompt,
                "raw_prompt": raw_prompt,
                "negative_prompt": neg_prompt,
                "style_preset": selected_style,
                "width": images[0].size[0],
                "height": images[0].size[1],
                "reference_photo": Path(ref_photos[0]).name,
                "seed": current_seed,
                "generation_time_seconds": round(duration, 3)
            }
        elif self.config.engine_mode == "openai_dalle3":
            images = []
            api_meta = {}
            if self.mock:
                images.append(Image.new("RGB", (1024, 1024), color=(20, 25, 35)))
                api_meta = {"provider_name": "OpenAI DALL-E 3 (Mock)"}
            else:
                for _ in range(num_images):
                    img, meta_data = DirectAPIProvider.generate_openai_dalle3(
                        prompt=pos_prompt,
                        api_key=self.config.api_key,
                    )
                    images.append(img)
                    api_meta.update(meta_data)
            duration = time.time() - start_time
            meta = {
                "engine_mode": self.config.engine_mode,
                "api_provider": "openai_dalle3",
                "base_model": "OpenAI DALL-E 3 (ChatGPT)",
                "prompt": pos_prompt,
                "raw_prompt": raw_prompt,
                "negative_prompt": neg_prompt,
                "style_preset": selected_style,
                "width": images[0].size[0],
                "height": images[0].size[1],
                "seed": current_seed,
                "generation_time_seconds": round(duration, 3)
            }
        else:
            # Local Diffusion Path (SDXL LoRA or Fast Instant FaceID + LCM)
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

