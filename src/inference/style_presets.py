"""
Photographic and artistic style presets for SDXL LoRA inference.
Defines positive stylistic enhancements and specialized negative filters
for diverse genres and camera profiles.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class StylePreset:
    name: str
    display_name: str
    description: str
    prompt_suffix: str
    negative_suffix: str


PRESETS: Dict[str, StylePreset] = {
    "candid_smartphone": StylePreset(
        name="candid_smartphone",
        display_name="Candid Smartphone Photo",
        description="Authentic iPhone/Pixel snapshot with natural skin texture and real-life ambient light",
        prompt_suffix=(
            ", candid authentic smartphone photograph, shot on iPhone 15 Pro, natural skin pores and micro-textures, "
            "ambient room lighting, casual framing, photorealistic, realistic skin tone, real life snapshot"
        ),
        negative_suffix=(
            ", studio lights, professional flash, airbrush, photoshop, beauty filter, plastic skin, cartoon, anime, "
            "illustration, 3d render, CGI, glossy, oversaturated, neon outlines, sketch, drawing, painting, art"
        )
    ),
    "photorealistic": StylePreset(
        name="photorealistic",
        display_name="Photorealistic Portrait",
        description="Crisp 85mm portrait photography with natural skin texture and depth of field",
        prompt_suffix=(
            ", shot on 85mm f/1.4 lens, natural skin pores and micro-textures, photorealistic, "
            "subsurface scattering, 8k uhd, soft studio rim lighting, perfectly focused"
        ),
        negative_suffix=(
            ", plastic skin, cartoon, anime, illustration, airbrushed, 3d render, oversaturated, CGI, "
            "sketch, drawing, neon outlines, comic book, line art"
        )
    ),
    "cinematic": StylePreset(
        name="cinematic",
        display_name="Cinematic 35mm Film",
        description="Hollywood film still with anamorphic framing and atmospheric lighting",
        prompt_suffix=(
            ", 35mm anamorphic film still, cinematic lighting, Kodak Vision3 color grading, "
            "atmospheric haze, shallow depth of field, dramatic shadows, blockbuster look"
        ),
        negative_suffix=(
            ", cheap video, amateur snapshot, blown out highlights, flat lighting, digital noise"
        )
    ),
    "studio_portrait": StylePreset(
        name="studio_portrait",
        display_name="High-End Studio Headshot",
        description="Clean commercial studio portrait with softbox rim lighting",
        prompt_suffix=(
            ", professional corporate studio headshot, softbox lighting, catchlight in eyes, "
            "neutral backdrop, sharp focus on face, Vogue magazine cover quality, 8k"
        ),
        negative_suffix=(
            ", harsh flash, cluttered background, informal, noisy, blurry, pixelated"
        )
    ),
    "cyberpunk": StylePreset(
        name="cyberpunk",
        display_name="Cyberpunk Neon Night",
        description="Futuristic neon-lit night cityscape with vibrant color contrasts",
        prompt_suffix=(
            ", cyberpunk night aesthetic, vibrant neon blue and magenta lighting, reflective wet asphalt, "
            "holographic glow, futuristic urban atmosphere, high contrast cinematic"
        ),
        negative_suffix=(
            ", daytime, sunny, boring colors, washed out, monochromatic, pastoral"
        )
    ),
    "vintage_polaroid": StylePreset(
        name="vintage_polaroid",
        display_name="Vintage 90s Polaroid",
        description="Nostalgic instant film photograph with authentic tones",
        prompt_suffix=(
            ", authentic 1990s polaroid snapshot, warm nostalgic color tones, gentle flash glow, "
            "subtle vintage grain, soft vignettes, authentic retro aesthetic"
        ),
        negative_suffix=(
            ", modern digital camera, HDR, overly sharp, 4k render, glossy"
        )
    ),
    "black_and_white": StylePreset(
        name="black_and_white",
        display_name="Dramatic Monochrome",
        description="Fine-art monochrome portrait with rich silver gelatin contrast",
        prompt_suffix=(
            ", dramatic black and white fine art portrait, Hasselblad medium format, rich tonality, "
            "deep blacks, pure whites, silver gelatin print look, iconic portraiture"
        ),
        negative_suffix=(
            ", color, sepia, washed out grays, digital artifacting, muddy shadows"
        )
    ),
    "oil_painting": StylePreset(
        name="oil_painting",
        display_name="Classical Oil Painting",
        description="Masterpiece oil portrait with classical Rembrandt chiaroscuro",
        prompt_suffix=(
            ", oil on canvas portrait in the style of classical masters, visible impasto brushstrokes, "
            "Rembrandt lighting, rich painterly textures, museum masterpiece"
        ),
        negative_suffix=(
            ", photograph, 3d render, vector, digital illustration, flat art"
        )
    )
}


def get_style_preset(name: Optional[str]) -> Optional[StylePreset]:
    """Retrieves a style preset by name (case-insensitive)."""
    if not name:
        return None
    return PRESETS.get(name.lower().strip())


def list_available_styles() -> List[Dict[str, str]]:
    """Returns a list of all registered styles with names and descriptions."""
    return [
        {
            "name": p.name,
            "display_name": p.display_name,
            "description": p.description
        }
        for p in PRESETS.values()
    ]

