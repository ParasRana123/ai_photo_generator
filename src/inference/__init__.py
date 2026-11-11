"""
Inference pipeline module for Phase 2: Personal AI Photo Generator.
Handles SDXL model loading, LoRA adapter attachment, prompt styling,
and image generation with embedded metadata.
"""

from .pipeline_loader import SDXLPipelineLoader, MockSDXLInferencePipeline
from .style_presets import StylePreset, PRESETS, get_style_preset, list_available_styles
from .prompt_enhancer import PromptEnhancer

__all__ = [
    "SDXLPipelineLoader",
    "MockSDXLInferencePipeline",
    "StylePreset",
    "PRESETS",
    "get_style_preset",
    "list_available_styles",
    "PromptEnhancer"
]
