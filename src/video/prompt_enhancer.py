"""
Temporal prompt enhancer and anti-flicker filter for Text-to-Video generation.
Injects fluid motion physics, temporal continuity cues, and eliminates video jitter.
"""

from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class VideoStylePreset:
    name: str
    display_name: str
    description: str
    motion_prompt_suffix: str
    negative_suffix: str


VIDEO_PRESETS: Dict[str, VideoStylePreset] = {
    "cinematic_movie": VideoStylePreset(
        name="cinematic_movie",
        display_name="Cinematic 35mm Motion",
        description="Hollywood film camera motion with depth of field, natural motion blur, and film grain",
        motion_prompt_suffix=(
            ", cinematic movie motion, 35mm film aesthetic, smooth steadycam tracking shot, "
            "natural motion blur, photorealistic lighting transitions, continuous fluid movement"
        ),
        negative_suffix=(
            ", flickering, jitter, sudden morphing, temporal jitter, stutter, warped faces, static freeze"
        )
    ),
    "slow_motion": VideoStylePreset(
        name="slow_motion",
        display_name="Ultra Slow Motion",
        description="High frame rate slow-motion capturing micro-expressions and fluid dynamics",
        motion_prompt_suffix=(
            ", captured in 120fps ultra slow motion, fluid physics, detailed micro-movements, "
            "smooth continuous temporal interpolation, crisp focus, studio lighting"
        ),
        negative_suffix=(
            ", fast jerking motion, speed ramp glitches, frame drops, low frame rate, ghosting"
        )
    ),
    "drone_aerial": VideoStylePreset(
        name="drone_aerial",
        display_name="Drone Aerial Sweep",
        description="Expansive aerial sweep with gimbal stabilization and wide dynamic range",
        motion_prompt_suffix=(
            ", smooth aerial drone cinematography, 4k gimbal stabilized flight, sweeping vista movement, "
            "gradual perspective shift, realistic atmospheric parallax"
        ),
        negative_suffix=(
            ", camera shake, sudden rotation, disorienting jump cuts, pixelated horizon"
        )
    ),
    "hyperlapse": VideoStylePreset(
        name="hyperlapse",
        display_name="City Hyperlapse",
        description="Dynamic motion timelapse with flowing light trails and accelerated scenery",
        motion_prompt_suffix=(
            ", dynamic smooth hyperlapse motion, continuous forward camera path, moving clouds, "
            "flowing light streaks, stabilized long exposure motion"
        ),
        negative_suffix=(
            ", jerky stops, irregular cadence, blurred trails, erratic shaking"
        )
    ),
    "candid_vlog": VideoStylePreset(
        name="candid_vlog",
        display_name="Candid Smartphone Video",
        description="Authentic natural camera movement with casual framing and natural skin tones",
        motion_prompt_suffix=(
            ", candid smartphone 4k 60fps video, shot on iPhone 15 Pro, natural handheld breathing motion, "
            "authentic ambient lighting, realistic unedited skin texture and facial movement"
        ),
        negative_suffix=(
            ", artificial 3d animation, CGI, plastic waxy skin, robotic movement, cartoon"
        )
    )
}


class VideoPromptEnhancer:
    """Enhances text prompts for high-fidelity temporal consistency and fluid video motion."""

    DEFAULT_NEGATIVE_VIDEO = (
        "flickering, jitter, temporal discontinuity, frame morphing, warped faces, deformed hands, "
        "sudden character transformation, static slide show, stop motion stutter, low frame rate, "
        "blurry, low resolution, watermark, text, video compression artifacts"
    )

    def __init__(self, default_style: str = "cinematic_movie"):
        self.default_style = default_style

    def build_video_prompts(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style_name: Optional[str] = None,
        camera_instruction: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Combines base prompt, style motion cues, camera directives, and anti-flicker filters.
        """
        clean_prompt = " ".join(prompt.split()).strip().rstrip(",")

        # 1. Apply style preset
        chosen_style = style_name or self.default_style
        preset = VIDEO_PRESETS.get(chosen_style.lower().strip())
        if preset is not None:
            clean_prompt = f"{clean_prompt}{preset.motion_prompt_suffix}"
            base_neg = negative_prompt.strip() if negative_prompt and negative_prompt.strip() else self.DEFAULT_NEGATIVE_VIDEO
            final_neg = f"{base_neg}{preset.negative_suffix}"
        else:
            final_neg = negative_prompt.strip() if negative_prompt and negative_prompt.strip() else self.DEFAULT_NEGATIVE_VIDEO

        # 2. Append camera instruction if provided
        if camera_instruction:
            clean_prompt = f"{clean_prompt}, {camera_instruction.strip()}"

        clean_prompt = " ".join(clean_prompt.split()).strip().rstrip(",")
        final_neg = " ".join(final_neg.split()).strip().rstrip(",")

        return clean_prompt, final_neg

    @staticmethod
    def list_presets() -> list:
        return [
            {
                "name": p.name,
                "display_name": p.display_name,
                "description": p.description,
            }
            for p in VIDEO_PRESETS.values()
        ]
