"""
Smart reference photo selector for Free Cloud ZeroGPU personalization.
Scans 10-15 uploaded photos, scores facial visibility and photographic sharpness,
ranks the sharpest portraits, and persists a lightweight cloud profile JSON in ~1-2 seconds.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageOps

from src.preprocessing.face_detector import FaceDetector


class ReferencePhotoSelector:
    """Evaluates and ranks user photos to pick optimal identity references for cloud models."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, use_face_detection: bool = True):
        self.face_detector = FaceDetector(min_face_size=(48, 48)) if use_face_detection else None

    @staticmethod
    def _compute_sharpness(image: Image.Image) -> float:
        """Computes Laplacian variance sharpness score on resized gray thumbnail."""
        thumb = image.copy()
        thumb.thumbnail((512, 512), Image.Resampling.BILINEAR)
        gray = np.array(thumb.convert("L"))
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def score_photo(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Scores an individual photo based on face presence, face prominence, and sharpness."""
        p = Path(image_path)
        img = Image.open(p)
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size

        sharpness = self._compute_sharpness(img)
        face_rect = self.face_detector.detect_primary_face(img) if self.face_detector else None

        face_area_ratio = 0.0
        center_bonus = 0.0
        if face_rect is not None:
            fx, fy, fw, fh = face_rect
            face_area_ratio = (fw * fh) / max(1, w * h)
            cx = (fx + fw / 2.0) / max(1, w)
            cy = (fy + fh / 2.0) / max(1, h)
            dist = ((cx - 0.5) ** 2 + (cy - 0.45) ** 2) ** 0.5
            center_bonus = max(0.0, 1.0 - dist * 2.0)

        # Weighted score favoring clear visible centered faces + crisp focus
        total_score = (
            (100.0 if face_rect is not None else 0.0)
            + min(sharpness, 400.0) * 0.25
            + (face_area_ratio * 200.0)
            + (center_bonus * 40.0)
        )

        return {
            "path": str(p),
            "filename": p.name,
            "width": w,
            "height": h,
            "face_detected": face_rect is not None,
            "sharpness": round(sharpness, 2),
            "score": round(total_score, 2)
        }

    def build_profile(
        self,
        input_dir: Union[str, Path] = "data/raw",
        output_path: Union[str, Path] = "checkpoints/cloud_face_profile.json",
        trigger_word: str = "sks person",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        mock: bool = False
    ) -> Dict[str, Any]:
        """
        Ranks all photos in input_dir and saves the top reference photos to a JSON profile.
        """
        start_time = time.time()
        in_dir = Path(input_dir)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not in_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {in_dir}")

        all_images = sorted([
            p for p in in_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ])
        if not all_images:
            raise ValueError(f"No photos found in '{in_dir}'. Upload photos first.")

        # Prefer real user photos over generated sample_photo_* files when both exist
        user_images = [p for p in all_images if not p.name.startswith("sample_photo_")]
        scan_files = user_images if user_images else all_images

        scored: List[Dict[str, Any]] = []
        for idx, p in enumerate(scan_files):
            if mock:
                scored.append({
                    "path": str(p),
                    "filename": p.name,
                    "width": 1024,
                    "height": 1024,
                    "face_detected": True,
                    "sharpness": 150.0,
                    "score": 150.0 - idx
                })
            else:
                scored.append(self.score_photo(p))

            if progress_callback:
                progress_callback(idx + 1, len(scan_files))

        ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
        top_photos = [item["path"] for item in ranked[:4]]
        duration = round(time.time() - start_time, 3)

        profile = {
            "primary_photo": top_photos[0],
            "top_photos": top_photos,
            "total_photos_analyzed": len(ranked),
            "faces_detected": sum(1 for item in ranked if item["face_detected"]),
            "trigger_word": trigger_word,
            "duration_seconds": duration,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ranked": ranked
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

        return profile

    @staticmethod
    def load_profile(profile_path: Union[str, Path] = "checkpoints/cloud_face_profile.json") -> Optional[Dict[str, Any]]:
        """Loads a saved cloud face profile JSON if it exists and references valid files."""
        p = Path(profile_path)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("primary_photo") and Path(data["primary_photo"]).exists():
                return data
        except Exception:
            return None
        return None

