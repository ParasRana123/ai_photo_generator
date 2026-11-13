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
        """
        Scores an individual photo based on face resolution, face-crop sharpness,
        and penalizes low-res phone screenshots with black/UI bars.
        """
        p = Path(image_path)
        img = Image.open(p)
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size

        img_gray = np.array(img.convert("L"))
        face_rect = self.face_detector.detect_primary_face(img) if self.face_detector else None

        face_sharpness = 0.0
        face_width = 0
        is_screenshot = False
        if face_rect is not None:
            fx, fy, fw, fh = face_rect
            face_width = fw
            face_region = img_gray[fy:fy+fh, fx:fx+fw]
            if face_region.size > 0:
                face_sharpness = float(cv2.Laplacian(face_region, cv2.CV_64F).var())

            # Detect phone screenshots with top/bottom black borders or UI headers
            top_dark = float(img_gray[:40, :].mean()) < 35.0
            bot_dark = float(img_gray[-40:, :].mean()) < 35.0
            is_screenshot = top_dark and bot_dark
        else:
            face_sharpness = self._compute_sharpness(img)

        # High score for sharp, large faces from real cameras; heavily penalize low-res screenshots
        total_score = (
            (150.0 if face_rect is not None else 0.0)
            + min(face_sharpness, 1000.0) * 0.4
            + min(face_width, 400.0) * 0.6
            - (250.0 if is_screenshot else 0.0)
        )

        return {
            "path": str(p),
            "filename": p.name,
            "width": w,
            "height": h,
            "face_detected": face_rect is not None,
            "sharpness": round(face_sharpness, 2),
            "is_screenshot": is_screenshot,
            "score": round(total_score, 2)
        }

    def prepare_portrait_crop(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        target_size: tuple = (1024, 1024),
        padding_scale: float = 2.6,
    ) -> str:
        """
        Crops a clean, face-centered portrait (default 1024x1024) with generous headroom so
        cloud models do not inherit extreme phone aspect ratios (e.g. 448x1024) or low-res faces.
        """
        p = Path(image_path)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        w, h = img.size

        face_rect = self.face_detector.detect_primary_face(img) if self.face_detector else None
        if face_rect is not None:
            fx, fy, fw, fh = face_rect
            cx = fx + fw // 2
            cy = fy + fh // 2
            box = int(max(fw, fh) * padding_scale)
            box = min(box, min(w, h))
            half = box // 2
            x1 = max(0, min(w - box, cx - half))
            y1 = max(0, min(h - box, cy - int(box * 0.42)))
            x2 = min(w, x1 + box)
            y2 = min(h, y1 + box)
        else:
            side = min(w, h)
            x1 = (w - side) // 2
            y1 = max(0, (h - side) // 3)
            x2 = x1 + side
            y2 = y1 + side

        cropped = img.crop((x1, y1, x2, y2)).resize(target_size, Image.Resampling.LANCZOS)
        cropped.save(out, format="JPEG", quality=95)
        return str(out)

    def build_profile(
        self,
        input_dir: Union[str, Path] = "data/raw",
        output_path: Union[str, Path] = "checkpoints/cloud_face_profile.json",
        trigger_word: str = "sks person",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        mock: bool = False
    ) -> Dict[str, Any]:
        """
        Ranks all photos in input_dir and saves the top reference photos + clean square portraits
        to a JSON profile.
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

        prepared_photos: List[str] = []
        if not mock:
            crops_dir = out_file.parent / "cloud_portraits"
            for i, photo_path in enumerate(top_photos):
                crop_out = crops_dir / f"portrait_{i}.jpg"
                try:
                    prepared_photos.append(self.prepare_portrait_crop(photo_path, crop_out))
                except Exception:
                    prepared_photos.append(photo_path)
        else:
            prepared_photos = list(top_photos)

        duration = round(time.time() - start_time, 3)

        profile = {
            "primary_photo": prepared_photos[0],
            "top_photos": prepared_photos,
            "raw_top_photos": top_photos,
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

