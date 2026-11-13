"""
Direct Commercial & Cloud API Providers for AI Photo Generation.
Supports:
  1. Google Imagen 3 (Gemini's Image Model via Google AI Studio API)
  2. fal.ai (FLUX.1-dev + PuLID on Enterprise H100 GPUs)
  3. OpenAI DALL-E 3
"""

import base64
import json
import logging
import os
import time
import urllib.request
import urllib.error
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

logger = logging.getLogger("APIProviders")


class DirectAPIProvider:
    """Dispatches generation requests to Google Gemini/Imagen 3, fal.ai, or OpenAI."""

    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @classmethod
    def generate_gemini_imagen(
        cls,
        prompt: str,
        api_key: Optional[str] = None,
        aspect_ratio: str = "1:1",
        num_images: int = 1,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generates photorealistic portraits using Google's Imagen 3 API
        (the exact foundation model powering Google Gemini).
        """
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key is required. Get a free key at https://aistudio.google.com/app/apikey "
                "and pass it in UI or set GEMINI_API_KEY in your environment."
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={key.strip()}"
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": min(max(num_images, 1), 4),
                "aspectRatio": aspect_ratio,
                "personGeneration": "ALLOW_ADULT",
                "outputMimeType": "image/jpeg"
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            raise RuntimeError(f"Google Imagen 3 API Error ({e.code}): {err_msg}")

        predictions = data.get("predictions")
        if not predictions or "bytesBase64Encoded" not in predictions[0]:
            raise RuntimeError(f"Unexpected Imagen 3 response payload: {data}")

        img_bytes = base64.b64decode(predictions[0]["bytesBase64Encoded"])
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return img, {
            "api_provider": "google_imagen3",
            "model": "imagen-3.0-generate-002",
            "provider_name": "Google Gemini (Imagen 3)"
        }

    @classmethod
    def generate_fal_ai(
        cls,
        prompt: str,
        reference_photo: Optional[str] = None,
        api_key: Optional[str] = None,
        steps: int = 28,
        guidance_scale: float = 4.0,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generates ultra-fast personal portraits on fal.ai using FLUX.1-dev PuLID.
        """
        key = api_key or os.getenv("FAL_KEY")
        if not key:
            raise ValueError(
                "fal.ai API key is required. Get your key at https://fal.ai/dashboard/keys "
                "and pass it in UI or set FAL_KEY in your environment."
            )

        # Submit to fal.ai FLUX PuLID queue
        url = "https://queue.fal.run/fal-ai/flux-pulid"
        headers = {
            "Authorization": f"Key {key.strip()}",
            "Content-Type": "application/json"
        }

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        }

        if reference_photo and Path(reference_photo).exists():
            b64_data = cls._image_to_base64(reference_photo)
            payload["reference_image_url"] = f"data:image/jpeg;base64,{b64_data}"

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            raise RuntimeError(f"fal.ai API submission error ({e.code}): {err_msg}")

        # If synchronous result
        if "images" in data and data["images"]:
            img_url = data["images"][0]["url"]
        elif "request_id" in data:
            # Poll status queue
            req_id = data["request_id"]
            status_url = f"https://queue.fal.run/fal-ai/flux-pulid/requests/{req_id}"
            start = time.time()
            img_url = None
            while time.time() - start < 60:
                time.sleep(1.0)
                poll_req = urllib.request.Request(status_url, headers=headers)
                with urllib.request.urlopen(poll_req, timeout=20) as p_resp:
                    p_data = json.loads(p_resp.read().decode("utf-8"))
                    if p_data.get("status") == "COMPLETED":
                        img_url = p_data["response"]["images"][0]["url"]
                        break
            if not img_url:
                raise TimeoutError("fal.ai generation timed out.")
        else:
            raise RuntimeError(f"Unexpected fal.ai response: {data}")

        with urllib.request.urlopen(img_url, timeout=30) as img_resp:
            img = Image.open(BytesIO(img_resp.read())).convert("RGB")

        return img, {
            "api_provider": "fal_ai",
            "model": "fal-ai/flux-pulid",
            "provider_name": "fal.ai (FLUX.1-dev H100)"
        }

    @classmethod
    def generate_openai_dalle3(
        cls,
        prompt: str,
        api_key: Optional[str] = None,
        size: str = "1024x1024",
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generates realistic images using OpenAI DALL-E 3 API (ChatGPT's image generator).
        """
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key is required. Get your key at https://platform.openai.com/api-keys "
                "and pass it in UI or set OPENAI_API_KEY in your environment."
            )

        url = "https://api.openai.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {key.strip()}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "dall-e-3",
            "prompt": prompt,
            "n": 1,
            "size": size,
            "quality": "hd",
            "style": "natural"
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            raise RuntimeError(f"OpenAI DALL-E 3 API Error ({e.code}): {err_msg}")

        img_url = data["data"][0]["url"]
        with urllib.request.urlopen(img_url, timeout=30) as img_resp:
            img = Image.open(BytesIO(img_resp.read())).convert("RGB")

        return img, {
            "api_provider": "openai_dalle3",
            "model": "dall-e-3",
            "provider_name": "OpenAI DALL-E 3 (ChatGPT)"
        }

