"""
Caption generation and prompt template engine for personal LoRA training.
Injects trigger words and creates paired captions with diverse contextual phrasing.
"""

from pathlib import Path
from typing import List, Optional, Union
import random


class CaptionGenerator:
    """Generates trigger-word infused captions for image fine-tuning."""

    DEFAULT_TEMPLATES = [
        "a photo of {trigger}",
        "a close up portrait photo of {trigger}",
        "a high resolution photo of {trigger} looking directly at the camera",
        "a headshot photo of {trigger} with neutral expression",
        "a portrait photograph of {trigger} in natural lighting",
        "a clear photograph of {trigger} face",
        "a professional photo of {trigger}",
        "a candid shot of {trigger}"
    ]

    def __init__(
        self,
        trigger_word: str = "sks person",
        templates: Optional[List[str]] = None,
        seed: Optional[int] = 42
    ):
        self.trigger_word = trigger_word
        self.templates = templates or self.DEFAULT_TEMPLATES
        self.rng = random.Random(seed)

    def generate_caption(self, index: int, custom_caption: Optional[str] = None) -> str:
        """
        Generate a caption for the given image index.
        If a custom caption is provided, ensures the trigger word is included.
        """
        if custom_caption and custom_caption.strip():
            cleaned = custom_caption.strip()
            if self.trigger_word.lower() not in cleaned.lower():
                return f"{self.trigger_word}, {cleaned}"
            return cleaned

        template = self.templates[index % len(self.templates)]
        return template.format(trigger=self.trigger_word)

    def create_caption_file(
        self,
        image_path: Union[str, Path],
        caption: str,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """Saves a paired .txt caption file corresponding to an image."""
        img_p = Path(image_path)
        dest_dir = Path(output_dir) if output_dir else img_p.parent
        caption_path = dest_dir / f"{img_p.stem}.txt"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption.strip() + "\n")
        return caption_path
