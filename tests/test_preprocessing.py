"""Unit tests for image preprocessing, face detection, and caption generation."""

import pytest
from PIL import Image
from src.preprocessing.image_processor import ImageProcessor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.caption_generator import CaptionGenerator
from src.preprocessing.dataset_builder import DatasetBuilder


def test_image_processor_crop_and_resize():
    processor = ImageProcessor(target_resolution=1024, use_face_detection=False)
    # Create landscape test image
    img = Image.new("RGB", (1600, 1200), color=(100, 150, 200))
    resized, meta = processor.crop_and_resize(img)

    assert resized.size == (1152, 896) or resized.size == (1024, 1024)
    assert meta["crop_method"] == "center_crop"
    assert "blur_score" in meta


def test_image_processor_exif_and_alpha():
    processor = ImageProcessor(use_face_detection=False)
    # Create RGBA image with transparency
    rgba_img = Image.new("RGBA", (800, 800), color=(255, 0, 0, 128))
    converted, _ = processor.crop_and_resize(rgba_img.convert("RGB"))
    assert converted.mode == "RGB"
    assert converted.size == (1024, 1024)


def test_caption_generator_templates():
    gen = CaptionGenerator(trigger_word="john doe")
    cap1 = gen.generate_caption(0)
    assert "john doe" in cap1

    custom = gen.generate_caption(1, "walking in the park")
    assert "john doe" in custom
    assert "walking in the park" in custom


def test_dataset_builder_validation(tmp_path):
    builder = DatasetBuilder(input_dir=tmp_path, min_photos=10, max_photos=15)
    # Test with 5 photos (under minimum 10)
    fake_paths = [tmp_path / f"photo_{i}.png" for i in range(5)]
    status = builder.validate_dataset(fake_paths)
    assert not status["meets_phase1_target"]
    assert len(status["warnings"]) > 0

    # Test with 12 photos (within 10-15 target)
    valid_paths = [tmp_path / f"photo_{i}.png" for i in range(12)]
    valid_status = builder.validate_dataset(valid_paths)
    assert valid_status["meets_phase1_target"]
    assert len(valid_status["warnings"]) == 0

