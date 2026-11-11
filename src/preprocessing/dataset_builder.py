"""
Dataset builder and validation manager.
Validates the photo count requirement (10-15 photos for Phase 1), processes images,
generates paired captions, and exports metadata.jsonl for training.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import logging

from .image_processor import ImageProcessor
from .caption_generator import CaptionGenerator

logger = logging.getLogger("DatasetBuilder")


class DatasetBuilder:
    """Orchestrates photo ingestion, validation, processing, and pairing."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        input_dir: Union[str, Path] = "data/raw",
        output_dir: Union[str, Path] = "data/processed",
        trigger_word: str = "sks person",
        target_resolution: int = 1024,
        min_photos: int = 10,
        max_photos: int = 30,
        use_face_detection: bool = True
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.trigger_word = trigger_word
        self.target_resolution = target_resolution
        self.min_photos = min_photos
        self.max_photos = max_photos

        self.processor = ImageProcessor(
            target_resolution=target_resolution,
            use_face_detection=use_face_detection
        )
        self.captioner = CaptionGenerator(trigger_word=trigger_word)

    def scan_raw_images(self) -> List[Path]:
        """Discovers all valid image files in the input directory."""
        if not self.input_dir.exists():
            return []

        images = [
            p for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        return sorted(images)

    def validate_dataset(self, raw_images: List[Path]) -> Dict[str, Any]:
        """
        Validates photo count against Phase 1 requirements (recommended 10-15 photos).
        Returns diagnostic status and warning flags.
        """
        count = len(raw_images)
        status = {
            "total_images": count,
            "is_valid": count >= 1,
            "meets_phase1_target": self.min_photos <= count <= self.max_photos,
            "warnings": []
        }

        if count < self.min_photos:
            status["warnings"].append(
                f"Phase 1 recommends at least {self.min_photos} photos for optimal facial LoRA fidelity (found {count})."
            )
        elif count > self.max_photos:
            status["warnings"].append(
                f"Found {count} photos. More than {self.max_photos} photos may cause overfitting or extended training times."
            )

        return status

    def build(self) -> Dict[str, Any]:
        """
        Executes the full preprocessing pipeline:
        1. Scan & validate input directory
        2. Normalizes, crops, and resizes photos
        3. Generates trigger captions & metadata.jsonl
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_images = self.scan_raw_images()
        validation = self.validate_dataset(raw_images)

        processed_records: List[Dict[str, Any]] = []
        jsonl_path = self.output_dir / "metadata.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
            for idx, img_path in enumerate(raw_images):
                dest_filename = f"image_{idx:03d}.png"
                dest_path = self.output_dir / dest_filename

                # Read custom caption if accompanying .txt exists
                custom_txt = img_path.with_suffix(".txt")
                custom_caption = None
                if custom_txt.exists():
                    try:
                        with open(custom_txt, "r", encoding="utf-8") as f:
                            custom_caption = f.read().strip()
                    except Exception:
                        pass

                # Preprocess image
                meta = self.processor.process_file(img_path, dest_path, save_format="PNG")

                # Generate and write caption
                caption = self.captioner.generate_caption(idx, custom_caption)
                self.captioner.create_caption_file(dest_path, caption)

                record = {
                    "file_name": dest_filename,
                    "text": caption,
                    "target_size": meta["target_size"],
                    "crop_method": meta["crop_method"],
                    "blur_score": meta["blur_score"],
                    "face_detected": meta["face_detected"]
                }
                processed_records.append(record)
                jsonl_file.write(json.dumps(record) + "\n")

        summary = {
            "validation": validation,
            "processed_count": len(processed_records),
            "output_directory": str(self.output_dir),
            "metadata_jsonl": str(jsonl_path),
            "records": processed_records
        }
        return summary
