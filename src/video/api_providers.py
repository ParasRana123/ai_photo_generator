"""
Commercial API Video Providers (fal.ai Kling/Minimax and Google Veo).
Connects to high-throughput cloud video rendering clusters via REST APIs.
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("VideoAPIProvider")


class VideoAPIProvider:
    """Dispatches text-to-video requests to fal.ai or Google Veo."""

    @classmethod
    def generate_fal_video(
        cls,
        prompt: str,
        model: str = "fal-ai/kling-video/v1/standard/text-to-video",
        api_key: Optional[str] = None,
        duration: float = 5.0,
        aspect_ratio: str = "16:9",
        output_dir: str = "outputs/videos",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generates video using fal.ai (Kling 1.5, Minimax Video-01, or Luma Dream Machine).
        """
        key = api_key or os.getenv("FAL_KEY")
        if not key:
            raise ValueError(
                "fal.ai API key is required. Obtain a key at https://fal.ai/dashboard/keys "
                "and pass it in UI or set FAL_KEY in your environment."
            )

        endpoint_url = f"https://queue.fal.run/{model}"
        headers = {
            "Authorization": f"Key {key.strip()}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "duration": str(int(duration)),
            "aspect_ratio": aspect_ratio,
        }

        req = urllib.request.Request(
            endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            raise RuntimeError(f"fal.ai video queue error ({e.code}): {err_msg}")

        # Check synchronous or poll queue
        video_url = None
        if "video" in data and isinstance(data["video"], dict) and "url" in data["video"]:
            video_url = data["video"]["url"]
        elif "request_id" in data:
            req_id = data["request_id"]
            status_url = f"https://queue.fal.run/{model}/requests/{req_id}"
            start = time.time()
            while time.time() - start < 180:
                time.sleep(2.0)
                poll_req = urllib.request.Request(status_url, headers=headers)
                try:
                    with urllib.request.urlopen(poll_req, timeout=20) as p_resp:
                        p_data = json.loads(p_resp.read().decode("utf-8"))
                        if p_data.get("status") == "COMPLETED":
                            video_url = p_data.get("response", {}).get("video", {}).get("url")
                            break
                        elif p_data.get("status") == "FAILED":
                            raise RuntimeError(f"fal.ai video generation failed: {p_data}")
                except Exception:
                    pass
            if not video_url:
                raise TimeoutError("fal.ai video generation timed out after 180s.")
        else:
            raise RuntimeError(f"Unexpected fal.ai video response: {data}")

        # Download video to local output_dir
        dest = Path(output_dir) / f"fal_video_{int(time.time())}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(video_url, timeout=60) as v_resp:
            dest.write_bytes(v_resp.read())

        return str(dest), {
            "provider": "fal_ai",
            "model": model,
            "duration_seconds": duration,
            "source_url": video_url,
        }

    @classmethod
    def generate_veo_video(
        cls,
        prompt: str,
        api_key: Optional[str] = None,
        duration: float = 5.0,
        output_dir: str = "outputs/videos",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generates video using Google Veo 2 / Gemini Video generation.
        """
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key is required for Google Veo. Obtain a key at "
                "https://aistudio.google.com/app/apikey and pass it in UI or environment."
            )

        # Direct Google AI Studio Veo Video Generation REST endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/veo-2.0-generate-001:predict?key={key.strip()}"
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "durationSeconds": int(duration),
                "aspectRatio": "16:9",
                "personGeneration": "ALLOW_ADULT",
            }
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            raise RuntimeError(f"Google Veo API error ({e.code}): {err_msg}")

        dest = Path(output_dir) / f"veo_video_{int(time.time())}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Handle predictions bytes or videoUri
        preds = data.get("predictions", [])
        if preds and "bytesBase64Encoded" in preds[0]:
            import base64
            dest.write_bytes(base64.b64decode(preds[0]["bytesBase64Encoded"]))
        elif preds and "videoUri" in preds[0]:
            with urllib.request.urlopen(preds[0]["videoUri"], timeout=60) as v_resp:
                dest.write_bytes(v_resp.read())
        else:
            raise RuntimeError(f"Unexpected Google Veo response payload: {data}")

        return str(dest), {
            "provider": "google_veo",
            "model": "veo-2.0-generate-001",
            "duration_seconds": duration,
        }
