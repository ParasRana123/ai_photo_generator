"""
Face detector module for portrait framing and smart auto-cropping.
Uses OpenCV Haar cascade classifiers with graceful fallback to center framing.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image


class FaceDetector:
    """Detects facial bounding boxes in images to guide smart crop centering."""

    def __init__(self, min_face_size: Tuple[int, int] = (64, 64)):
        self.min_face_size = min_face_size
        self.face_cascade = None
        try:
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if hasattr(cv2, "CascadeClassifier"):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    def detect_primary_face(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the most prominent face in an image.
        Returns:
            (x, y, w, h) bounding box of the largest face, or None if no face detected.
        """
        if self.face_cascade is None or self.face_cascade.empty():
            return None

        try:
            np_img = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            equalized = cv2.equalizeHist(gray)

            faces = self.face_cascade.detectMultiScale(
                equalized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 0:
                return None

            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return tuple(int(v) for v in largest_face)
        except Exception:
            return None

    def compute_face_centered_crop(
        self,
        image_width: int,
        image_height: int,
        face_rect: Tuple[int, int, int, int],
        target_aspect_ratio: float,
        padding_ratio: float = 0.35
    ) -> Tuple[int, int, int, int]:
        """
        Calculates crop coordinates (left, top, right, bottom) centered around the face
        while matching the target aspect ratio (width / height).
        """
        fx, fy, fw, fh = face_rect
        face_center_x = fx + fw / 2.0
        face_center_y = fy + fh / 2.0

        # Portrait composition: position face slightly above center (rule of thirds)
        face_center_y = max(fh / 2.0, face_center_y - (fh * 0.1))

        current_aspect = image_width / image_height

        if current_aspect > target_aspect_ratio:
            crop_h = image_height
            crop_w = int(crop_h * target_aspect_ratio)
            left = int(face_center_x - crop_w / 2.0)
            left = max(0, min(left, image_width - crop_w))
            top = 0
            right = left + crop_w
            bottom = crop_h
        else:
            crop_w = image_width
            crop_h = int(crop_w / target_aspect_ratio)
            top = int(face_center_y - crop_h / 2.0)
            top = max(0, min(top, image_height - crop_h))
            left = 0
            right = crop_w
            bottom = top + crop_h

        return left, top, right, bottom
