"""
Image processing pipeline for personal photos.
Handles EXIF auto-orientation, color space normalization, face-aware smart cropping,
aspect ratio bucketing, and Lanczos resizing for SDXL training.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
import cv2
import numpy as np
from PIL import Image, ImageOps

from .face_detector import FaceDetector


class ImageProcessor:
    """Preprocesses user photos into high-quality standardized SDXL training images."""

    DEFAULT_BUCKETS = [
        (1024, 1024),  # 1:1 square
        (896, 1152),   # 3:4 portrait
        (1152, 896),   # 4:3 landscape
        (768, 1344),   # 9:16 tall portrait
        (1344, 768),   # 16:9 widescreen
    ]

    def __init__(
        self,
        target_resolution: int = 1024,
        buckets: Optional[List[Tuple[int, int]]] = None,
        use_face_detection: bool = True,
        face_padding: float = 0.35
    ):
        self.target_resolution = target_resolution
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self.use_face_detection = use_face_detection
        self.face_padding = face_padding
        self.face_detector = FaceDetector() if use_face_detection else None

    def load_and_orient(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Loads an image from path and rotates according to EXIF metadata.
        Converts transparency/alpha channels cleanly to pure RGB on white canvas.
        """
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            alpha_composite = Image.alpha_composite(background, img)
            return alpha_composite.convert("RGB")

        return img.convert("RGB")

    def find_best_bucket(self, width: int, height: int) -> Tuple[int, int]:
        """Finds the SDXL bucket that minimizes crop loss for the given aspect ratio."""
        aspect = width / height
        best_bucket = self.buckets[0]
        min_diff = float("inf")

        for bw, bh in self.buckets:
            bucket_aspect = bw / bh
            diff = abs(aspect - bucket_aspect)
            if diff < min_diff:
                min_diff = diff
                best_bucket = (bw, bh)

        return best_bucket

    def compute_blur_score(self, image: Image.Image) -> float:
        """
        Estimates sharpness using the variance of the Laplacian filter.
        Values below 80-100 typically indicate noticeable blurriness.
        """
        np_img = np.array(image.convert("L"))
        variance = cv2.Laplacian(np_img, cv2.CV_64F).var()
        return float(variance)

    def crop_and_resize(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Processes a single PIL image:
        1. Determines target dimensions
        2. Detects face (if enabled) to center the crop around the subject
        3. Crops to exact aspect ratio
        4. High-quality Lanczos resizes to target resolution
        """
        orig_w, orig_h = image.size
        if target_size is None:
            target_size = self.find_best_bucket(orig_w, orig_h)

        target_w, target_h = target_size
        target_aspect = target_w / target_h

        face_rect = None
        if self.use_face_detection and self.face_detector is not None:
            face_rect = self.face_detector.detect_primary_face(image)

        if face_rect is not None:
            left, top, right, bottom = self.face_detector.compute_face_centered_crop(
                orig_w, orig_h, face_rect, target_aspect, self.face_padding
            )
            crop_method = "face_aware"
        else:
            # Fallback to standard center crop
            current_aspect = orig_w / orig_h
            if current_aspect > target_aspect:
                crop_w = int(orig_h * target_aspect)
                left = (orig_w - crop_w) // 2
                top = 0
                right = left + crop_w
                bottom = orig_h
            else:
                crop_h = int(orig_w / target_aspect)
                left = 0
                top = (orig_h - crop_h) // 2
                right = orig_w
                bottom = top + crop_h
            crop_method = "center_crop"

        cropped_img = image.crop((left, top, right, bottom))
        resized_img = cropped_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        blur_score = self.compute_blur_score(resized_img)

        metadata = {
            "original_size": (orig_w, orig_h),
            "target_size": (target_w, target_h),
            "crop_box": (left, top, right, bottom),
            "crop_method": crop_method,
            "face_detected": face_rect is not None,
            "blur_score": round(blur_score, 2),
            "is_sharp": blur_score > 80.0
        }

        return resized_img, metadata

    def process_file(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        target_size: Optional[Tuple[int, int]] = None,
        save_format: str = "PNG"
    ) -> Dict[str, Any]:
        """Loads an image file, normalizes, crops, resizes, and saves."""
        img = self.load_and_orient(image_path)
        processed, meta = self.crop_and_resize(img, target_size)

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        processed.save(out_p, format=save_format, quality=95)

        meta["source_path"] = str(image_path)
        meta["output_path"] = str(output_path)
        return meta

