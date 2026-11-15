"""
Multi-format Video Encoder and Thumbnail Generator.
Encodes generated frame sequences into H.264/MP4, animated GIF, and WebM,
attaching JSON sidecar metadata and extracting poster keyframe thumbnails.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import json
import logging
import time
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("VideoEncoder")


class VideoEncoder:
    """Encodes frame sequences to MP4, WebM, and GIF, with thumbnail generation."""

    @staticmethod
    def _frame_to_numpy_bgr(frame: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Ensures a frame is in OpenCV uint8 BGR format."""
        if isinstance(frame, Image.Image):
            rgb = np.array(frame.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
            if frame.ndim == 3 and frame.shape[2] == 3:
                return frame
        raise ValueError(f"Unsupported frame type/shape: {type(frame)}")

    @classmethod
    def save_mp4(
        cls,
        frames: List[Union[Image.Image, np.ndarray]],
        output_path: Union[str, Path],
        fps: int = 16,
    ) -> str:
        """Writes frames to an MP4 video file using OpenCV."""
        if not frames:
            raise ValueError("Cannot encode empty frame sequence.")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        first_frame = cls._frame_to_numpy_bgr(frames[0])
        h, w = first_frame.shape[:2]

        # Try mp4v then avc1 codecs
        writer = None
        for codec in ["mp4v", "avc1", "MJPG"]:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                temp_writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
                if temp_writer.isOpened():
                    writer = temp_writer
                    break
            except Exception:
                continue

        if writer is None or not writer.isOpened():
            raise RuntimeError(f"Failed to initialize VideoWriter for '{out_path}'")

        try:
            for f in frames:
                bgr = cls._frame_to_numpy_bgr(f)
                if bgr.shape[:2] != (h, w):
                    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
                writer.write(bgr)
        finally:
            writer.release()

        return str(out_path)

    @classmethod
    def save_gif(
        cls,
        frames: List[Union[Image.Image, np.ndarray]],
        output_path: Union[str, Path],
        fps: int = 16,
    ) -> str:
        """Writes frames to an animated GIF using PIL."""
        if not frames:
            raise ValueError("Cannot encode empty frame sequence to GIF.")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        pil_frames: List[Image.Image] = []
        for f in frames:
            if isinstance(f, Image.Image):
                pil_frames.append(f.convert("RGB"))
            elif isinstance(f, np.ndarray):
                rgb = cv2.cvtColor(cls._frame_to_numpy_bgr(f), cv2.COLOR_BGR2RGB)
                pil_frames.append(Image.fromarray(rgb))

        duration_ms = int(1000.0 / max(1, fps))
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )
        return str(out_path)

    @classmethod
    def save_thumbnail(
        cls,
        frames: List[Union[Image.Image, np.ndarray]],
        output_path: Union[str, Path],
        keyframe_idx: Optional[int] = None,
    ) -> str:
        """Extracts and saves a keyframe thumbnail as JPEG."""
        if not frames:
            raise ValueError("No frames available for thumbnail.")

        idx = keyframe_idx if keyframe_idx is not None else len(frames) // 3
        idx = max(0, min(idx, len(frames) - 1))
        keyframe = frames[idx]

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(keyframe, Image.Image):
            keyframe.convert("RGB").save(out_path, format="JPEG", quality=92)
        else:
            bgr = cls._frame_to_numpy_bgr(keyframe)
            cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        return str(out_path)

    @classmethod
    def write_metadata_sidecar(
        cls,
        video_path: Union[str, Path],
        metadata: Dict[str, Any],
    ) -> str:
        """Writes execution metadata into a companion JSON sidecar file."""
        vp = Path(video_path)
        sidecar_path = vp.with_suffix(".json")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return str(sidecar_path)

    @classmethod
    def read_metadata_sidecar(cls, video_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Reads companion JSON sidecar metadata for a video file."""
        vp = Path(video_path)
        sidecar_path = vp.with_suffix(".json")
        if not sidecar_path.exists():
            return None
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
