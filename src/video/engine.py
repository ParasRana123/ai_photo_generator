"""
Core Text-to-Video Inference Engine.
Orchestrates prompt building, camera motion conditioning, multi-provider routing,
frame synthesis, MP4/GIF encoding, thumbnail generation, and metadata embedding.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import time
import logging
from PIL import Image

from src.utils.video_config import VideoConfig
from .prompt_enhancer import VideoPromptEnhancer
from .camera_motion import CameraMotionController
from .encoder import VideoEncoder
from .cloud_client import CloudZeroGPUVideoClient
from .api_providers import VideoAPIProvider

logger = logging.getLogger("TextToVideoEngine")


@dataclass
class VideoGenerationResult:
    """Encapsulates generated video file path, thumbnail, and execution metadata."""
    video_path: str
    thumbnail_path: str
    gif_path: Optional[str]
    prompt: str
    negative_prompt: str
    seed: int
    fps: int
    num_frames: int
    duration_seconds: float
    metadata: Dict[str, Any]
    generation_time_seconds: float


class TextToVideoEngine:
    """High-level generator interface for Text-to-Video synthesis."""

    def __init__(
        self,
        config: Optional[VideoConfig] = None,
        mock: bool = False,
    ):
        self.config = config or VideoConfig()
        self.mock = mock

        self.enhancer = VideoPromptEnhancer()
        self.motion_ctrl = CameraMotionController()
        self.cloud_client = CloudZeroGPUVideoClient(hf_token=self.config.hf_token)

    def generate(
        self,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        camera_motion: Optional[str] = None,
        motion_intensity: Optional[float] = None,
        duration_seconds: Optional[float] = None,
        fps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        provider: Optional[str] = None,
        seed: Optional[int] = None,
        style_preset: Optional[str] = "cinematic_movie",
    ) -> VideoGenerationResult:
        """
        Executes text-to-video synthesis pipeline and saves video, thumbnail, and sidecar JSON.
        """
        start_time = time.time()

        # 1. Resolve parameters
        raw_prompt = prompt or self.config.prompt
        raw_neg = negative_prompt or self.config.negative_prompt
        selected_motion = camera_motion or self.config.camera_motion
        selected_intensity = motion_intensity or self.config.motion_intensity
        selected_provider = provider or self.config.provider
        w = width or self.config.width
        h = height or self.config.height
        vid_fps = fps or self.config.fps
        dur = duration_seconds or self.config.duration_seconds
        current_seed = seed if seed is not None else (self.config.seed or int(time.time()) % 100000)

        num_frames = int(vid_fps * dur)
        num_frames = max(8, min(num_frames, 120))

        # 2. Build camera motion instruction and enhance prompt
        cam_instruction = self.motion_ctrl.get_prompt_descriptor(selected_motion, selected_intensity)
        pos_prompt, neg_prompt = self.enhancer.build_video_prompts(
            raw_prompt,
            negative_prompt=raw_neg,
            style_name=style_preset,
            camera_instruction=cam_instruction,
        )

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"video_{timestamp}_{current_seed}"

        mp4_path = str(out_dir / f"{base_filename}.mp4")
        thumb_path = str(out_dir / f"{base_filename}_thumb.jpg")
        gif_path = str(out_dir / f"{base_filename}.gif") if self.config.generate_gif_preview else None

        execution_meta: Dict[str, Any] = {
            "provider": selected_provider,
            "raw_prompt": raw_prompt,
            "prompt": pos_prompt,
            "negative_prompt": neg_prompt,
            "camera_motion": selected_motion,
            "motion_intensity": selected_intensity,
            "duration_seconds": dur,
            "fps": vid_fps,
            "num_frames": num_frames,
            "width": w,
            "height": h,
            "seed": current_seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # 3. Dispatch to chosen video provider
        if selected_provider in ("fal_kling", "fal_minimax"):
            fal_model = (
                "fal-ai/minimax/video-01/text-to-video"
                if selected_provider == "fal_minimax"
                else "fal-ai/kling-video/v1/standard/text-to-video"
            )
            if self.mock:
                frames = CloudZeroGPUVideoClient.generate_mock_video_frames(
                    prompt=pos_prompt, width=w, height=h, num_frames=num_frames,
                    camera_motion=selected_motion, provider=selected_provider, seed=current_seed
                )
                VideoEncoder.save_mp4(frames, mp4_path, fps=vid_fps)
                VideoEncoder.save_thumbnail(frames, thumb_path)
                if gif_path:
                    VideoEncoder.save_gif(frames, gif_path, fps=vid_fps)
            else:
                out_file, meta = VideoAPIProvider.generate_fal_video(
                    prompt=pos_prompt, model=fal_model, api_key=self.config.api_key,
                    duration=dur, output_dir=str(out_dir)
                )
                mp4_path = out_file
                execution_meta.update(meta)

        elif selected_provider == "google_veo":
            if self.mock:
                frames = CloudZeroGPUVideoClient.generate_mock_video_frames(
                    prompt=pos_prompt, width=w, height=h, num_frames=num_frames,
                    camera_motion=selected_motion, provider=selected_provider, seed=current_seed
                )
                VideoEncoder.save_mp4(frames, mp4_path, fps=vid_fps)
                VideoEncoder.save_thumbnail(frames, thumb_path)
                if gif_path:
                    VideoEncoder.save_gif(frames, gif_path, fps=vid_fps)
            else:
                out_file, meta = VideoAPIProvider.generate_veo_video(
                    prompt=pos_prompt, api_key=self.config.api_key,
                    duration=dur, output_dir=str(out_dir)
                )
                mp4_path = out_file
                execution_meta.update(meta)

        else:
            # Free Cloud ZeroGPU / Mock Engine
            output_or_frames, meta = self.cloud_client.generate_video(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                provider=selected_provider,
                width=w,
                height=h,
                num_frames=num_frames,
                camera_motion=selected_motion,
                seed=current_seed,
                mock=self.mock or selected_provider == "mock"
            )
            execution_meta.update(meta)

            if isinstance(output_or_frames, list):
                # We have a sequence of PIL images -> encode to MP4, thumbnail, and optional GIF
                VideoEncoder.save_mp4(output_or_frames, mp4_path, fps=vid_fps)
                VideoEncoder.save_thumbnail(output_or_frames, thumb_path)
                if gif_path:
                    VideoEncoder.save_gif(output_or_frames, gif_path, fps=vid_fps)
            elif isinstance(output_or_frames, str):
                # Video file downloaded from Gradio space
                mp4_path = output_or_frames

        # Generate thumbnail from MP4 if not created yet
        if not Path(thumb_path).exists():
            import cv2
            cap = cv2.VideoCapture(mp4_path)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(thumb_path, frame)
            cap.release()

        gen_duration = round(time.time() - start_time, 3)
        execution_meta["generation_time_seconds"] = gen_duration

        # 4. Save metadata sidecar
        VideoEncoder.write_metadata_sidecar(mp4_path, execution_meta)

        return VideoGenerationResult(
            video_path=mp4_path,
            thumbnail_path=thumb_path,
            gif_path=gif_path,
            prompt=pos_prompt,
            negative_prompt=neg_prompt,
            seed=current_seed,
            fps=vid_fps,
            num_frames=num_frames,
            duration_seconds=dur,
            metadata=execution_meta,
            generation_time_seconds=gen_duration,
        )
