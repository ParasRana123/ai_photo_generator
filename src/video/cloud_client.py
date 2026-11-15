"""
Free Cloud GPU Video Client (Zero-Cost Text-to-Video).
Routes text-to-video prompts to live, free Hugging Face ZeroGPU Spaces
(THUDM/CogVideoX-5B, ModelScope Damo T2V, AnimateDiff) via gradio_client
with automatic multi-provider failover.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import time
import urllib.request
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger("CloudVideoClient")


class CloudZeroGPUVideoClient:
    """Client for zero-cost cloud GPU Text-to-Video generation via HuggingFace Spaces."""

    VIDEO_SPACES = {
        "cogvideox": "THUDM/CogVideoX-5B-Space",
        "damo_t2v": "ali-vilab/text-to-video-synthesis",
        "animatediff": "guoyww/AnimateDiff",
    }

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token.strip() if hf_token and hf_token.strip() else None

    @staticmethod
    def _extract_video_path(item: Any, output_dir: Union[str, Path] = "outputs/videos") -> str:
        """Extracts or downloads video file from a Gradio return payload."""
        path_or_url = None
        if isinstance(item, str):
            path_or_url = item
        elif isinstance(item, dict):
            if "video" in item and isinstance(item["video"], dict):
                path_or_url = item["video"].get("path") or item["video"].get("url")
            else:
                path_or_url = item.get("path") or item.get("url") or item.get("video")

        if not path_or_url:
            raise ValueError(f"Unexpected Gradio video payload: {item}")

        if str(path_or_url).startswith(("http://", "https://")):
            dest = Path(output_dir) / f"cloud_t2v_{int(time.time())}.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(path_or_url, timeout=90) as resp:
                dest.write_bytes(resp.read())
            return str(dest)

        return str(path_or_url)

    @staticmethod
    def generate_mock_video_frames(
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_frames: int = 24,
        camera_motion: str = "static",
        provider: str = "mock",
        seed: int = 42,
    ) -> List[Image.Image]:
        """
        Synthesizes deterministic mock video frames with animated visual patterns
        and motion indicators for offline testing and fast verification.
        """
        frames: List[Image.Image] = []
        for i in range(num_frames):
            t = i / max(1, num_frames - 1)
            img = Image.new("RGB", (width, height), color=(18, 22, 34))
            draw = ImageDraw.Draw(img)

            # Draw moving cinematic horizon & ambient glow
            r_val = int(30 + 50 * np.sin(t * np.pi))
            g_val = int(50 + 60 * np.cos(t * np.pi))
            b_val = int(120 + 80 * np.sin(t * np.pi * 0.5))
            draw.rectangle([0, int(height * 0.6), width, height], fill=(r_val, g_val, b_val))

            # Draw moving animated subject orb with motion trajectory
            cx = int(width * (0.35 + 0.3 * np.sin(t * 2 * np.pi)))
            cy = int(height * (0.45 + 0.1 * np.cos(t * 2 * np.pi)))
            radius = int(width * 0.14 * (1.0 + 0.15 * t if "zoom" in camera_motion else 1.0))
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(235, 180, 110), outline=(255, 255, 255), width=2)

            # Metadata tags overlay
            draw.text((25, 25), f"[Mock T2V] {provider.upper()} | Frame {i+1}/{num_frames}", fill=(255, 255, 255))
            draw.text((25, 45), f"Motion: {camera_motion} | Seed: {seed}", fill=(180, 210, 240))
            draw.text((25, height - 35), f"Prompt: {prompt[:50]}...", fill=(200, 220, 200))

            frames.append(img)
        return frames

    def _run_cogvideox(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
    ) -> Tuple[str, Dict[str, Any]]:
        from gradio_client import Client
        client = Client(self.VIDEO_SPACES["cogvideox"], token=self.hf_token, verbose=False)
        res = client.predict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            api_name="/predict"
        )
        video_path = self._extract_video_path(res[0] if isinstance(res, (list, tuple)) else res)
        return video_path, {"space": self.VIDEO_SPACES["cogvideox"], "model": "THUDM/CogVideoX-5B"}

    def _run_damo_t2v(
        self,
        prompt: str,
        seed: int,
    ) -> Tuple[str, Dict[str, Any]]:
        from gradio_client import Client
        client = Client(self.VIDEO_SPACES["damo_t2v"], token=self.hf_token, verbose=False)
        res = client.predict(
            prompt,
            int(seed),
            api_name="/run"
        )
        video_path = self._extract_video_path(res[0] if isinstance(res, (list, tuple)) else res)
        return video_path, {"space": self.VIDEO_SPACES["damo_t2v"], "model": "ModelScope/Damo-T2V"}

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        provider: str = "auto",
        width: int = 512,
        height: int = 512,
        num_frames: int = 24,
        camera_motion: str = "static",
        seed: Optional[int] = None,
        mock: bool = False,
    ) -> Tuple[Union[str, List[Image.Image]], Dict[str, Any]]:
        """
        Dispatches Text-to-Video generation to free Hugging Face Spaces or mock engine.
        Returns either an encoded video path (str) or a list of frames (List[Image.Image]).
        """
        actual_seed = seed if seed is not None else int(time.time()) % 100000

        if mock:
            frames = self.generate_mock_video_frames(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                camera_motion=camera_motion,
                provider=provider,
                seed=actual_seed
            )
            return frames, {
                "provider": provider,
                "space": f"mock/{provider}",
                "model": "Mock Video Generator",
                "is_mock": True
            }

        chain = ["cogvideox", "damo_t2v"] if provider == "auto" else [provider]
        errors: List[str] = []

        for candidate in chain:
            try:
                logger.info(f"Dispatching video generation to '{candidate}'...")
                if candidate == "cogvideox":
                    vpath, meta = self._run_cogvideox(prompt, negative_prompt, actual_seed)
                elif candidate == "damo_t2v":
                    vpath, meta = self._run_damo_t2v(prompt, actual_seed)
                else:
                    raise ValueError(f"Unsupported free video provider: {candidate}")

                meta["provider_used"] = candidate
                meta["is_mock"] = False
                return vpath, meta
            except Exception as exc:
                logger.warning(f"Free video space '{candidate}' error: {exc}")
                errors.append(f"{candidate}: {exc}")

        # If cloud spaces hit rate limit / quota, fall back cleanly to synthetic mock
        logger.warning(f"All free video spaces unavailable ({' | '.join(errors)}). Falling back to mock engine.")
        frames = self.generate_mock_video_frames(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            camera_motion=camera_motion,
            provider="fallback_mock",
            seed=actual_seed
        )
        return frames, {
            "provider": "fallback_mock",
            "space": "local/fallback_mock",
            "model": "Local Fallback Video Generator",
            "is_mock": True,
            "errors": errors
        }
