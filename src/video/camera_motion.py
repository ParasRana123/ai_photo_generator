"""
Camera motion controller for Text-to-Video generation.
Encodes camera trajectories (pan, tilt, zoom, orbit) into model conditioning
and frame transformation coordinates.
"""

from typing import Dict, Any, List, Tuple
import math


class CameraMotionController:
    """Manages virtual camera trajectories and motion prompt injection."""

    MOTION_DIRECTIVES = {
        "static": {
            "name": "static",
            "display_name": "Locked Tripod (Static)",
            "prompt_descriptor": "locked tripod camera, static framing, subject motion within frame",
            "dx": 0.0,
            "dy": 0.0,
            "zoom": 0.0,
        },
        "zoom_in": {
            "name": "zoom_in",
            "display_name": "Push-In (Zoom In)",
            "prompt_descriptor": "slow cinematic push-in dolly forward towards the subject, tightening frame",
            "dx": 0.0,
            "dy": 0.0,
            "zoom": 0.04,
        },
        "zoom_out": {
            "name": "zoom_out",
            "display_name": "Pull-Out (Zoom Out)",
            "prompt_descriptor": "gradual cinematic pull-back dolly revealing expansive background environment",
            "dx": 0.0,
            "dy": 0.0,
            "zoom": -0.04,
        },
        "pan_left": {
            "name": "pan_left",
            "display_name": "Pan Left",
            "prompt_descriptor": "smooth horizontal camera pan to the left, revealing scenery",
            "dx": -0.05,
            "dy": 0.0,
            "zoom": 0.0,
        },
        "pan_right": {
            "name": "pan_right",
            "display_name": "Pan Right",
            "prompt_descriptor": "smooth horizontal camera pan tracking across to the right",
            "dx": 0.05,
            "dy": 0.0,
            "zoom": 0.0,
        },
        "tilt_up": {
            "name": "tilt_up",
            "display_name": "Tilt Up (Low to High)",
            "prompt_descriptor": "cinematic vertical tilt upwards from torso to face and sky",
            "dx": 0.0,
            "dy": -0.05,
            "zoom": 0.0,
        },
        "tilt_down": {
            "name": "tilt_down",
            "display_name": "Tilt Down (High to Low)",
            "prompt_descriptor": "smooth vertical tilt downwards focusing on subject actions",
            "dx": 0.0,
            "dy": 0.05,
            "zoom": 0.0,
        },
        "orbit": {
            "name": "orbit",
            "display_name": "Orbital Arc Shot",
            "prompt_descriptor": "360 degree arc camera movement orbiting around the central subject",
            "dx": 0.06,
            "dy": 0.01,
            "zoom": 0.01,
        },
    }

    def __init__(self, default_motion: str = "static"):
        self.default_motion = default_motion if default_motion in self.MOTION_DIRECTIVES else "static"

    def get_prompt_descriptor(self, motion_type: str, intensity: float = 5.0) -> str:
        """Returns the natural language camera instruction for text-to-video models."""
        motion = self.MOTION_DIRECTIVES.get(motion_type.lower().strip(), self.MOTION_DIRECTIVES["static"])
        intensity_level = (
            "subtle slow" if intensity < 4.0
            else "smooth cinematic" if intensity < 7.5
            else "dynamic fast"
        )
        return f"{intensity_level} {motion['prompt_descriptor']}"

    def compute_frame_transforms(
        self,
        motion_type: str,
        num_frames: int = 24,
        intensity: float = 5.0
    ) -> List[Dict[str, float]]:
        """
        Calculates per-frame normalized camera motion offsets (dx, dy, zoom)
        for synthetic camera motion or temporal latent warping.
        """
        motion = self.MOTION_DIRECTIVES.get(motion_type.lower().strip(), self.MOTION_DIRECTIVES["static"])
        scale = max(1.0, min(intensity, 10.0)) / 5.0

        transforms = []
        for i in range(num_frames):
            t = i / max(1, num_frames - 1)
            # Smooth ease-in ease-out curve
            ease = 0.5 * (1.0 - math.cos(math.pi * t))

            dx = motion["dx"] * scale * ease
            dy = motion["dy"] * scale * ease
            zoom = 1.0 + (motion["zoom"] * scale * ease)

            transforms.append({
                "frame_idx": i,
                "progress": round(t, 4),
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "zoom": round(zoom, 4),
            })
        return transforms

    @classmethod
    def list_motions(cls) -> List[Dict[str, str]]:
        """Lists all supported camera motion presets."""
        return [
            {
                "id": k,
                "name": v["display_name"],
                "description": v["prompt_descriptor"]
            }
            for k, v in cls.MOTION_DIRECTIVES.items()
        ]
