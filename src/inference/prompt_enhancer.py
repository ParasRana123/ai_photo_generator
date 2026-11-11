"""
Prompt enhancement and trigger token injector for SDXL LoRA inference.
Ensures subject trigger words are properly positioned and combines user inputs
with photographic style presets and negative prompts.
"""

from typing import Tuple, Optional
from .style_presets import get_style_preset, StylePreset


class PromptEnhancer:
    """Enhances raw prompts with trigger word positioning and photographic styles."""

    DEFAULT_NEGATIVE_BASE = (
        "blurry, bad quality, low resolution, distorted face, deformed eyes, extra limbs, ugly, "
        "poorly drawn hands, bad anatomy, missing fingers, cropped head, watermark, signature"
    )

    def __init__(
        self,
        trigger_word: str = "sks person",
        auto_inject_trigger: bool = True
    ):
        self.trigger_word = trigger_word.strip()
        self.auto_inject_trigger = auto_inject_trigger

    def inject_trigger_word(self, raw_prompt: str) -> str:
        """
        Ensures the trigger word is present in the prompt.
        If missing, intelligently prepends it to anchor the subject identity.
        """
        prompt = raw_prompt.strip()
        if not self.auto_inject_trigger or not self.trigger_word:
            return prompt

        if self.trigger_word.lower() in prompt.lower():
            return prompt

        # If user wrote "a portrait of a man...", replace or prefix
        return f"a photo of {self.trigger_word}, {prompt}"

    def build_prompts(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style_preset_name: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Assembles finalized positive and negative prompts:
        1. Enforces trigger word presence
        2. Applies style preset enhancements
        3. Merges and deduplicates negative prompts
        """
        # 1. Process trigger word
        final_prompt = self.inject_trigger_word(prompt)

        # 2. Base negative prompt
        base_neg = negative_prompt.strip() if negative_prompt and negative_prompt.strip() else self.DEFAULT_NEGATIVE_BASE
        final_neg = base_neg

        # 3. Apply style preset
        preset = get_style_preset(style_preset_name)
        if preset is not None:
            final_prompt = f"{final_prompt.rstrip(',')}{preset.prompt_suffix}"
            final_neg = f"{final_neg.rstrip(',')}{preset.negative_suffix}"

        # Clean whitespace and trailing punctuation
        final_prompt = " ".join(final_prompt.split()).strip().rstrip(",")
        final_neg = " ".join(final_neg.split()).strip().rstrip(",")

        return final_prompt, final_neg

