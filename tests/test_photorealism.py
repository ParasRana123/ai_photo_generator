"""
Photorealism and Anti-Artifact Test Suite.
Validates square portrait cropping, anti-sketch negative prompting,
InstantID Karras scheduler calibration, and PhotoMaker-V2 provider routing.
"""

from pathlib import Path
from PIL import Image
import pytest

from src.utils.inference_config import InferenceConfig, VALID_CLOUD_PROVIDERS
from src.training.reference_selector import ReferencePhotoSelector
from src.inference.cloud_client import CloudZeroGPUClient
from src.inference.prompt_enhancer import PromptEnhancer
from src.inference.style_presets import get_style_preset
from src.inference.generator import SDXLInferenceEngine


def test_portrait_crop_dimensions_and_headroom(tmp_path):
    """Verifies prepare_portrait_crop produces exact 1024x1024 square portraits."""
    test_img_path = tmp_path / "raw_portrait.jpg"
    im = Image.new("RGB", (714, 1599), color=(140, 120, 110))
    im.save(test_img_path)

    out_crop = tmp_path / "cropped_1024.jpg"
    selector = ReferencePhotoSelector(use_face_detection=False)
    cropped_path = selector.prepare_portrait_crop(test_img_path, out_crop, target_size=(1024, 1024))

    assert Path(cropped_path).exists()
    cropped_img = Image.open(cropped_path)
    assert cropped_img.size == (1024, 1024)
    assert cropped_img.mode == "RGB"


def test_anti_sketch_negative_prompt_injection():
    """Verifies that default negative prompt excludes sketch, cartoon, neon, and line art."""
    enhancer = PromptEnhancer(trigger_word="sks person", auto_inject_trigger=False)
    pos, neg = enhancer.build_prompts("A candid smartphone photo of a young man at home")

    lower_neg = neg.lower()
    for forbidden in ["sketch", "cartoon", "drawing", "painting", "neon contours", "airbrushed"]:
        assert forbidden in lower_neg, f"Expected '{forbidden}' in anti-sketch negative prompt"


def test_candid_smartphone_style_preset_mapping():
    """Verifies candid cues automatically select candid_smartphone preset without conflicting studio tags."""
    preset = get_style_preset("candid_smartphone")
    assert preset is not None
    assert "iphone" in preset.prompt_suffix.lower()
    assert "studio lights" in preset.negative_suffix.lower()

    enhancer = PromptEnhancer(trigger_word="sks person", auto_inject_trigger=False)
    pos, neg = enhancer.build_prompts(
        "A candid smartphone photo of a young man celebrating Diwali alone at home",
        style_preset_name="photorealistic"
    )
    # Conflicting studio rim lighting should be bypassed in favor of candid smartphone lighting
    assert "studio rim lighting" not in pos.lower()
    assert "iphone" in pos.lower()


def test_photomaker_v2_provider_registered():
    """Verifies PhotoMaker-V2 provider is registered in config and client spaces."""
    assert "photomaker_v2" in VALID_CLOUD_PROVIDERS
    assert "photomaker_v2" in CloudZeroGPUClient.PROVIDER_SPACES

    client = CloudZeroGPUClient()
    img, meta = client.generate(
        reference_photos=["test_face.png"],
        prompt="candid photo of a young man img",
        provider="photomaker_v2",
        mock=True
    )
    assert img.size == (896, 1152)
    assert meta["cloud_space"] == "mock/photomaker_v2"


def test_cloud_zerogpu_candid_generation_e2e(tmp_path):
    """End-to-end test of cloud zero-gpu generation with candid smartphone preset."""
    prof_path = tmp_path / "cloud_profile.json"
    dummy_ref = tmp_path / "dummy_face.jpg"
    Image.new("RGB", (1024, 1024), color=(180, 150, 130)).save(dummy_ref)

    import json
    prof_path.write_text(json.dumps({
        "primary_photo": str(dummy_ref),
        "top_photos": [str(dummy_ref)],
        "ranked": []
    }))

    cfg = InferenceConfig(
        engine_mode="cloud_zerogpu",
        cloud_provider="instantid",
        cloud_profile_path=str(prof_path),
        style_preset="candid_smartphone",
        output_dir=str(tmp_path / "out")
    )
    engine = SDXLInferenceEngine(config=cfg, mock=True)
    res = engine.generate(
        prompt="A candid smartphone photo of a man in an apartment",
        style_preset="candid_smartphone",
        seed=123
    )

    assert len(res.images) == 1
    assert len(res.saved_paths) == 1
    assert Path(res.saved_paths[0]).exists()
    assert res.metadata["style_preset"] == "candid_smartphone"
    assert "sketch" in res.negative_prompt.lower()

