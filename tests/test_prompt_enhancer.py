"""Unit tests for prompt enhancer and style presets."""

from src.inference.prompt_enhancer import PromptEnhancer
from src.inference.style_presets import get_style_preset, list_available_styles


def test_trigger_word_injection():
    enhancer = PromptEnhancer(trigger_word="sks person", auto_inject_trigger=True)

    # Missing trigger word: should prepend
    prompt_missing = "drinking espresso at a Parisian cafe"
    pos, _ = enhancer.build_prompts(prompt_missing)
    assert "sks person" in pos

    # Existing trigger word: should not duplicate
    prompt_present = "a full body shot of sks person walking in the snow"
    pos2, _ = enhancer.build_prompts(prompt_present)
    assert pos2.count("sks person") == 1


def test_style_presets_enhancement():
    enhancer = PromptEnhancer(trigger_word="sks person")

    pos, neg = enhancer.build_prompts(
        "walking through rain",
        style_preset_name="cinematic"
    )
    assert "anamorphic" in pos.lower()
    assert "kodak" in pos.lower()
    assert "amateur snapshot" in neg.lower()


def test_list_and_get_style_presets():
    styles = list_available_styles()
    assert len(styles) >= 7

    photo = get_style_preset("photorealistic")
    assert photo is not None
    assert photo.name == "photorealistic"

    invalid = get_style_preset("non_existent_preset")
    assert invalid is None

