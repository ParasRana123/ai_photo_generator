"""
Inference pipeline module for Phase 2: Personal AI Photo Generator.
Handles SDXL model loading, LoRA adapter attachment, prompt styling,
and image generation with embedded metadata.
"""

from .pipeline_loader import SDXLPipelineLoader, MockSDXLInferencePipeline

__all__ = ["SDXLPipelineLoader", "MockSDXLInferencePipeline"]

