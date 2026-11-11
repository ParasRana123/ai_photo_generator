"""
Curated test prompt suite for validating personal SDXL LoRA models.
Covers diverse photographic genres (studio, lifestyle, cinematic, artistic, outdoor)
to rigorously assess identity preservation and style adaptability.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TestPrompt:
    name: str
    category: str
    prompt: str
    negative_prompt: str
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    seed: int = 42


class PromptSuite:
    """Manages test prompt scenarios for evaluating the trained LoRA."""

    DEFAULT_NEGATIVE = (
        "blurry, bad quality, low resolution, distorted face, bad anatomy, deformed eyes, extra limbs, ugly, poorly drawn"
    )

    RAW_PROMPTS = [
        {
            "name": "01_professional_studio",
            "category": "Corporate / Business",
            "template": "A professional studio portrait of {trigger} in a sharp charcoal suit, crisp white shirt, neutral gray background, 85mm lens, sharp focus, 8k, cinematic studio lighting",
            "guidance_scale": 7.0,
            "steps": 30,
            "seed": 101
        },
        {
            "name": "02_candid_lifestyle",
            "category": "Casual / Lifestyle",
            "template": "A candid photograph of {trigger} smiling warmly at a rustic coffee shop, morning sunlight streaming through the window, shallow depth of field, authentic film grain",
            "guidance_scale": 7.5,
            "steps": 30,
            "seed": 202
        },
        {
            "name": "03_cyberpunk_cinematic",
            "category": "Cinematic / Stylized",
            "template": "An epic cinematic night portrait of {trigger} in a neon-lit cyberpunk city, vibrant magenta and cyan rim lighting, wet asphalt reflections, atmospheric mist",
            "guidance_scale": 8.0,
            "steps": 35,
            "seed": 303
        },
        {
            "name": "04_golden_hour_outdoor",
            "category": "Outdoor / Nature",
            "template": "A close-up headshot of {trigger} during sunset golden hour in an open park, warm soft backlighting, bokeh particles, detailed facial features",
            "guidance_scale": 7.5,
            "steps": 30,
            "seed": 404
        },
        {
            "name": "05_fine_art_oil_painting",
            "category": "Artistic / Painterly",
            "template": "An oil painting portrait of {trigger} in the style of Rembrandt, dramatic chiaroscuro lighting, rich brush strokes, classical museum masterpiece",
            "guidance_scale": 8.0,
            "steps": 35,
            "seed": 505
        }
    ]

    def __init__(self, trigger_word: str = "sks person", negative_prompt: Optional[str] = None):
        self.trigger_word = trigger_word
        self.negative_prompt = negative_prompt or self.DEFAULT_NEGATIVE

    def get_prompts(self) -> List[TestPrompt]:
        """Builds and returns the instantiated list of TestPrompt objects."""
        prompts = []
        for item in self.RAW_PROMPTS:
            formatted_prompt = item["template"].format(trigger=self.trigger_word)
            prompts.append(
                TestPrompt(
                    name=item["name"],
                    category=item["category"],
                    prompt=formatted_prompt,
                    negative_prompt=self.negative_prompt,
                    guidance_scale=item["guidance_scale"],
                    num_inference_steps=item["steps"],
                    seed=item["seed"]
                )
            )
        return prompts


def get_default_test_prompts(trigger_word: str = "sks person") -> List[TestPrompt]:
    """Helper to quickly obtain the standard prompt test suite."""
    return PromptSuite(trigger_word=trigger_word).get_prompts()
