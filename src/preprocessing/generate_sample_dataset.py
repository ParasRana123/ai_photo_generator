"""
Synthetic sample photo generator for testing Phase 1 AI training pipeline.
Creates 12 varied test images with realistic portrait geometries, lighting gradients,
and color profiles to validate the entire workflow without requiring real photos upfront.
"""

from pathlib import Path
from typing import List, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_sample_portraits(
    output_dir: Union[str, Path] = "data/raw",
    count: int = 12,
    base_size: tuple = (1024, 1024)
) -> List[Path]:
    """
    Generates synthetic sample images matching the Phase 1 target (10-15 photos).
    Each image contains a stylized head/shoulders portrait silhouette with varied
    palettes, lighting gradients, and camera angles.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    palettes = [
        # (background_top, background_bottom, skin_tone, clothing_color)
        ((40, 50, 75), (20, 25, 40), (220, 180, 150), (30, 30, 35)),       # Studio Dark
        ((220, 230, 240), (180, 190, 210), (210, 175, 145), (60, 90, 140)), # Daylight
        ((30, 30, 30), (15, 15, 15), (225, 185, 155), (180, 40, 40)),       # High Contrast
        ((240, 210, 180), (200, 160, 130), (230, 190, 160), (45, 65, 50)), # Golden Hour
        ((50, 40, 60), (25, 20, 30), (215, 170, 140), (120, 60, 90)),      # Neon Glow
        ((210, 220, 220), (170, 180, 180), (200, 160, 135), (25, 25, 30)), # Minimalist
    ]

    generated_paths: List[Path] = []
    width, height = base_size

    for i in range(count):
        bg_top, bg_bot, skin, cloth = palettes[i % len(palettes)]

        # 1. Gradient background
        bg_array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            factor = y / height
            r = int(bg_top[0] * (1 - factor) + bg_bot[0] * factor)
            g = int(bg_top[1] * (1 - factor) + bg_bot[1] * factor)
            b = int(bg_top[2] * (1 - factor) + bg_bot[2] * factor)
            bg_array[y, :] = [r, g, b]

        img = Image.fromarray(bg_array)
        draw = ImageDraw.Draw(img)

        # 2. Add slight angle variations per photo
        x_offset = int((i % 3 - 1) * 30)
        y_offset = int((i % 4 - 1.5) * 20)

        center_x = width // 2 + x_offset
        center_y = int(height * 0.42) + y_offset

        # 3. Draw stylized shoulders / clothing
        shoulder_width = int(width * 0.55)
        shoulder_top = int(height * 0.65) + y_offset
        draw.ellipse(
            [center_x - shoulder_width, shoulder_top, center_x + shoulder_width, height + 200],
            fill=cloth
        )

        # 4. Draw neck
        neck_w = int(width * 0.16)
        neck_h = int(height * 0.22)
        draw.rectangle(
            [center_x - neck_w // 2, center_y + int(height * 0.1), center_x + neck_w // 2, shoulder_top + 20],
            fill=tuple(max(0, c - 20) for c in skin)
        )

        # 5. Draw head / face
        head_radius_x = int(width * 0.19)
        head_radius_y = int(height * 0.24)
        draw.ellipse(
            [center_x - head_radius_x, center_y - head_radius_y, center_x + head_radius_x, center_y + head_radius_y],
            fill=skin
        )

        # 6. Draw eyes and facial structure
        eye_y = center_y - int(head_radius_y * 0.08)
        eye_spacing = int(head_radius_x * 0.48)
        eye_w = int(head_radius_x * 0.18)
        eye_h = int(head_radius_y * 0.10)

        # Eyes
        draw.ellipse([center_x - eye_spacing - eye_w, eye_y - eye_h, center_x - eye_spacing + eye_w, eye_y + eye_h], fill=(40, 30, 20))
        draw.ellipse([center_x + eye_spacing - eye_w, eye_y - eye_h, center_x + eye_spacing + eye_w, eye_y + eye_h], fill=(40, 30, 20))

        # Nose bridge
        nose_top = eye_y + 10
        nose_bottom = eye_y + int(head_radius_y * 0.35)
        draw.line([center_x, nose_top, center_x, nose_bottom], fill=tuple(max(0, c - 30) for c in skin), width=4)

        # Mouth
        mouth_y = nose_bottom + int(head_radius_y * 0.22)
        mouth_w = int(head_radius_x * 0.35)
        draw.line([center_x - mouth_w, mouth_y, center_x + mouth_w, mouth_y], fill=(170, 90, 80), width=5)

        # 7. Apply gentle photographic blur for anti-aliasing
        img = img.filter(ImageFilter.SMOOTH)

        file_path = out_dir / f"sample_photo_{i + 1:02d}.png"
        img.save(file_path, "PNG")
        generated_paths.append(file_path)

    return generated_paths


if __name__ == "__main__":
    paths = generate_sample_portraits("data/raw", count=12)
    print(f"Generated {len(paths)} sample photos in data/raw/")
