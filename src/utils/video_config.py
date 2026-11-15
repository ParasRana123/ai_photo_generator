"""
Configuration schema and validation for Text-to-Video generation pipeline.
Manages video resolution, frame rates, temporal consistency, camera motion,
model providers, and export codecs.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import yaml
import json

VALID_VIDEO_PROVIDERS = [
    "auto",
    "cogvideox",
    "damo_t2v",
    "fal_kling",
    "fal_minimax",
    "google_veo",
    "mock"
]

VALID_CAMERA_MOTIONS = [
    "static",
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "orbit"
]

VALID_VIDEO_FORMATS = ["mp4", "gif", "webm"]


@dataclass
class VideoConfig:
    """Configuration parameters for Text-to-Video generation."""

    prompt: str = "A young man in a navy suit walking down a modern illuminated city street at dusk, cinematic camera tracking"
    negative_prompt: str = (
        "blurry, jitter, flickering, sudden morphing, warped faces, deformed limbs, static image, "
        "low frame rate, stutter, low resolution, watermark, text"
    )

    # Provider and Model
    provider: str = "auto"
    base_model: str = "THUDM/CogVideoX-5B"
    hf_token: Optional[str] = None
    api_key: Optional[str] = None

    # Temporal & Motion Dynamics
    duration_seconds: float = 3.0
    fps: int = 16
    num_frames: int = 24
    motion_intensity: float = 5.0  # 1.0 (subtle) to 10.0 (high dynamic motion)
    camera_motion: str = "static"

    # Frame Dimensions & Quality
    width: int = 512
    height: int = 512
    num_inference_steps: int = 25
    guidance_scale: float = 7.0
    seed: Optional[int] = 42

    # Storage and Export
    output_dir: str = "outputs/videos"
    export_format: str = "mp4"
    generate_thumbnail: bool = True
    generate_gif_preview: bool = True

    def validate(self) -> None:
        """Validates all video parameters against operational constraints."""
        if self.provider not in VALID_VIDEO_PROVIDERS:
            raise ValueError(f"Invalid provider '{self.provider}'. Choose from {VALID_VIDEO_PROVIDERS}")
        if self.camera_motion not in VALID_CAMERA_MOTIONS:
            raise ValueError(f"Invalid camera_motion '{self.camera_motion}'. Choose from {VALID_CAMERA_MOTIONS}")
        if self.export_format.lower() not in VALID_VIDEO_FORMATS:
            raise ValueError(f"Invalid export_format '{self.export_format}'. Choose from {VALID_VIDEO_FORMATS}")
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise ValueError(f"Dimensions ({self.width}x{self.height}) must be divisible by 8.")
        if self.width < 256 or self.height < 256:
            raise ValueError(f"Dimensions ({self.width}x{self.height}) must be at least 256x256.")
        if self.fps < 6 or self.fps > 60:
            raise ValueError(f"FPS must be between 6 and 60, got {self.fps}")
        if self.num_frames < 8 or self.num_frames > 240:
            raise ValueError(f"num_frames must be between 8 and 240, got {self.num_frames}")
        if self.duration_seconds <= 0.0 or self.duration_seconds > 30.0:
            raise ValueError(f"duration_seconds must be between 0.1 and 30.0, got {self.duration_seconds}")
        if self.motion_intensity < 1.0 or self.motion_intensity > 10.0:
            raise ValueError(f"motion_intensity must be between 1.0 and 10.0, got {self.motion_intensity}")
        if self.guidance_scale < 1.0 or self.guidance_scale > 20.0:
            raise ValueError(f"guidance_scale must be between 1.0 and 20.0, got {self.guidance_scale}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "VideoConfig":
        """Loads video configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        payload = data.get("video", data)
        cfg = cls(**payload)
        cfg.validate()
        return cfg

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Saves current configuration to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump({"video": asdict(self)}, f, sort_keys=False)
