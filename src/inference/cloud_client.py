"""
Free Cloud GPU Inference Client (fal.ai Free Alternative).
Routes personalized face generation requests to live, zero-cost Hugging Face ZeroGPU
Spaces (yanze/PuLID-FLUX, InstantX/InstantID, yanze/PuLID) via gradio_client
with automatic multi-provider failover.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import time
import urllib.request
from PIL import Image, ImageDraw

logger = logging.getLogger("CloudZeroGPUClient")


class CloudZeroGPUClient:
    """Client for zero-cost cloud GPU face personalization via HuggingFace ZeroGPU Spaces."""

    PROVIDER_SPACES = {
        "pulid_flux": "yanze/PuLID-FLUX",
        "instantid": "InstantX/InstantID",
        "pulid_sdxl": "yanze/PuLID",
    }

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token.strip() if hf_token and hf_token.strip() else None

    @staticmethod
    def _extract_image_from_output(item: Any) -> Image.Image:
        """Converts a Gradio file/image dictionary, filepath, or URL into a PIL Image."""
        path_or_url = None
        if isinstance(item, str):
            path_or_url = item
        elif isinstance(item, dict):
            if "image" in item and isinstance(item["image"], dict):
                path_or_url = item["image"].get("path") or item["image"].get("url")
            else:
                path_or_url = item.get("path") or item.get("url")

        if not path_or_url:
            raise ValueError(f"Unexpected Gradio image payload: {item}")

        if str(path_or_url).startswith(("http://", "https://")):
            with urllib.request.urlopen(path_or_url, timeout=60) as resp:
                from io import BytesIO
                return Image.open(BytesIO(resp.read())).convert("RGB")

        return Image.open(path_or_url).convert("RGB")

    def _run_pulid_flux(
        self,
        photos: List[str],
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        id_weight: float,
        seed: int,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        from gradio_client import Client, handle_file

        client = Client(self.PROVIDER_SPACES["pulid_flux"], token=self.hf_token, verbose=False)
        res = client.predict(
            prompt=prompt,
            id_image=handle_file(photos[0]),
            start_step=0,
            guidance=float(min(max(guidance_scale, 1.5), 8.0)),
            seed=str(seed),
            true_cfg=1.0,
            width=int(min(max(width, 512), 1152)),
            height=int(min(max(height, 512), 1152)),
            num_steps=int(min(max(steps, 8), 28)),
            id_weight=float(min(max(id_weight, 0.2), 1.5)),
            neg_prompt=negative_prompt,
            timestep_to_start_cfg=1,
            max_sequence_length=256,
            api_name="/generate_image",
        )
        img = self._extract_image_from_output(res[0])
        return img, {"cloud_space": "yanze/PuLID-FLUX", "cloud_model": "FLUX.1-dev + PuLID"}

    def _run_instantid(
        self,
        photos: List[str],
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        id_weight: float,
        seed: int,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        from gradio_client import Client, handle_file

        client = Client(self.PROVIDER_SPACES["instantid"], token=self.hf_token, verbose=False)
        # Disable LCM and calibrate CFG/steps for full SDXL photorealism (prevents neon/sketch frying)
        cfg = float(min(max(guidance_scale, 3.5), 7.0)) if guidance_scale >= 2.0 else 5.0
        actual_steps = int(min(max(steps, 20), 40)) if steps >= 10 else 25
        identitynet_ratio = float(min(max(id_weight, 0.35), 1.0)) * 0.8
        adapter_ratio = 0.65

        res = client.predict(
            handle_file(photos[0]),
            None,
            prompt,
            negative_prompt,
            "(No style)",
            actual_steps,
            identitynet_ratio,
            adapter_ratio,
            0.0,
            0.0,
            [],  # Empty controlnet selection avoids projecting depth contour masks onto skin
            cfg,
            int(seed),
            "DPMSolverMultistepScheduler-Karras",
            False,  # enable_lcm=False eliminates CFG/LCM saturation blowout
            True,   # enhance_face_region
            api_name="/generate_image",
        )
        img = self._extract_image_from_output(res[0])
        return img, {"cloud_space": "InstantX/InstantID", "cloud_model": "SDXL + InstantID (Karras)"}

    def _run_pulid_sdxl(
        self,
        photos: List[str],
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        id_weight: float,
        seed: int,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        from gradio_client import Client, handle_file

        client = Client(self.PROVIDER_SPACES["pulid_sdxl"], token=self.hf_token, verbose=False)
        p0 = handle_file(photos[0])
        p1 = handle_file(photos[1]) if len(photos) > 1 else None
        p2 = handle_file(photos[2]) if len(photos) > 2 else None
        p3 = handle_file(photos[3]) if len(photos) > 3 else None

        # SDXL-Lightning runs cleanly at 4-8 steps (over-stepping degrades details)
        lightning_steps = 4 if steps > 12 else int(min(max(steps, 4), 8))
        res = client.predict(
            p0, p1, p2, p3,
            prompt,
            negative_prompt,
            1.2,
            lightning_steps,
            int(seed),
            1,
            int(min(max(height, 512), 1024)),
            int(min(max(width, 512), 1024)),
            float(min(max(id_weight, 0.35), 1.0)) * 0.8,
            "fidelity",
            False,
            api_name="/run",
        )
        gallery = res[0]
        img = self._extract_image_from_output(gallery[0])
        return img, {"cloud_space": "yanze/PuLID", "cloud_model": "SDXL-Lightning + PuLID"}

    @staticmethod
    def _generate_mock_cloud_image(
        width: int,
        height: int,
        prompt: str,
        provider: str,
        seed: int,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        img = Image.new("RGB", (width, height), color=(22, 26, 36))
        draw = ImageDraw.Draw(img)
        draw.rectangle([30, 30, width - 30, height - 30], outline=(99, 102, 241), width=3)
        draw.ellipse([width // 4, height // 6, 3 * width // 4, height // 2], fill=(210, 175, 145))
        draw.text((50, int(height * 0.65)), f"[Cloud ZeroGPU Mock] Provider: {provider}", fill=(255, 255, 255))
        draw.text((50, int(height * 0.72)), f"Prompt: {prompt[:80]}", fill=(180, 190, 210))
        draw.text((50, int(height * 0.78)), f"Seed: {seed}", fill=(120, 220, 160))
        return img, {"cloud_space": f"mock/{provider}", "cloud_model": "Mock Cloud ZeroGPU"}

    def generate(
        self,
        reference_photos: List[str],
        prompt: str,
        negative_prompt: str = "blurry, low quality, deformed, watermark",
        provider: str = "auto",
        width: int = 896,
        height: int = 1152,
        steps: int = 20,
        guidance_scale: float = 4.0,
        id_weight: float = 1.0,
        seed: Optional[int] = None,
        mock: bool = False,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generates a personalized image on Free HuggingFace ZeroGPU Cloud.
        If provider == 'auto', tries pulid_flux -> instantid -> pulid_sdxl on failure.
        """
        if not reference_photos:
            raise ValueError("At least one reference photo is required for Cloud ZeroGPU generation.")

        actual_seed = seed if seed is not None else int(time.time()) % 100000

        if mock:
            return self._generate_mock_cloud_image(
                width, height, prompt, provider if provider != "auto" else "pulid_flux", actual_seed
            )

        chain = (
            ["pulid_flux", "instantid", "pulid_sdxl"]
            if provider == "auto"
            else [provider]
        )

        errors: List[str] = []
        for candidate in chain:
            try:
                logger.info(f"Dispatching Cloud ZeroGPU request to provider '{candidate}'...")
                if candidate == "pulid_flux":
                    img, meta = self._run_pulid_flux(
                        reference_photos, prompt, negative_prompt, width, height,
                        steps, guidance_scale, id_weight, actual_seed
                    )
                elif candidate == "instantid":
                    img, meta = self._run_instantid(
                        reference_photos, prompt, negative_prompt,
                        steps, guidance_scale, id_weight, actual_seed
                    )
                elif candidate == "pulid_sdxl":
                    img, meta = self._run_pulid_sdxl(
                        reference_photos, prompt, negative_prompt, width, height,
                        steps, id_weight, actual_seed
                    )
                else:
                    raise ValueError(f"Unsupported cloud provider: {candidate}")

                meta["provider_used"] = candidate
                return img, meta

            except Exception as exc:
                msg = f"{candidate}: {exc}"
                logger.warning(f"Cloud provider '{candidate}' failed: {exc}")
                errors.append(msg)

        raise RuntimeError(
            "All Free Cloud ZeroGPU providers failed or hit temporary HuggingFace quota. "
            "Tip: Add a free HuggingFace token (HF_TOKEN) for 4x daily GPU quota. "
            f"Details: {' | '.join(errors)}"
        )

