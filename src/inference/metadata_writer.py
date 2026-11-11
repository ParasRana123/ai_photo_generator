"""
PNG Metadata embedding and extraction utility for generated images.
Writes generation parameters (prompt, seed, steps, LoRA scale, model) into PNG
chunks for full reproducibility, compatible with Automatic1111 and ComfyUI inspection.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import time
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class PNGMetadataWriter:
    """Embeds generation parameters into PNG files using standard iTXt/tEXt chunks."""

    @staticmethod
    def build_png_info(metadata: Dict[str, Any]) -> PngInfo:
        """Constructs a PIL PngInfo object from a metadata dictionary."""
        png_info = PngInfo()

        # Format standard Automatic1111/Civitai parameters text block
        prompt = metadata.get("prompt", "")
        negative_prompt = metadata.get("negative_prompt", "")
        steps = metadata.get("num_inference_steps", 30)
        sampler = metadata.get("sampler", "Euler a")
        cfg = metadata.get("guidance_scale", 7.0)
        seed = metadata.get("seed", 42)
        size = f"{metadata.get('width', 1024)}x{metadata.get('height', 1024)}"
        model = metadata.get("base_model", "SDXL")
        lora = metadata.get("lora_name", "None")
        lora_scale = metadata.get("lora_scale", 1.0)

        # Build Civitai-compatible parameters string
        civ_params = (
            f"{prompt}\n"
            f"Negative prompt: {negative_prompt}\n"
            f"Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg}, Seed: {seed}, "
            f"Size: {size}, Model: {model}, LoRA: {lora}, LoRA scale: {lora_scale}"
        )

        png_info.add_text("parameters", civ_params)

        # Add structured key-values
        for k, v in metadata.items():
            if isinstance(v, (dict, list)):
                png_info.add_text(k, json.dumps(v))
            else:
                png_info.add_text(k, str(v))

        png_info.add_text("generator", "ai_photo_generator_phase2")
        png_info.add_text("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

        return png_info

    @staticmethod
    def save_image_with_metadata(
        image: Image.Image,
        output_path: Union[str, Path],
        metadata: Dict[str, Any]
    ) -> Path:
        """Saves a PIL image with embedded PNG generation metadata."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        png_info = PNGMetadataWriter.build_png_info(metadata)
        image.save(out_p, format="PNG", pnginfo=png_info, quality=95)
        return out_p

    @staticmethod
    def read_metadata(image_path: Union[str, Path]) -> Dict[str, Any]:
        """Extracts embedded metadata dictionary from a PNG image file."""
        img = Image.open(image_path)
        info = img.text or {}
        return dict(info)

