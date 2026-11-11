"""
PyTorch Dataset for SDXL LoRA fine-tuning.
Loads preprocessed images, paired text captions, prepares normalized tensors,
and tokenizes prompts for dual SDXL text encoders without requiring torchvision.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class SDXLDataset(Dataset):
    """
    Dataset for SDXL LoRA training with paired images and text prompts.
    Handles dual-tokenizer tokenization and image tensor normalization.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/processed",
        tokenizer_1: Optional[Any] = None,
        tokenizer_2: Optional[Any] = None,
        max_token_length: int = 77,
        target_resolution: int = 1024,
        trigger_word: str = "sks person"
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.max_token_length = max_token_length
        self.target_resolution = target_resolution
        self.trigger_word = trigger_word

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Loads paired image and caption files from the processed directory."""
        samples = []
        if not self.data_dir.exists():
            return samples

        # Check for metadata.jsonl first
        jsonl_path = self.data_dir / "metadata.jsonl"
        if jsonl_path.exists():
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line.strip())
                            img_path = self.data_dir / record.get("file_name", "")
                            if img_path.exists():
                                samples.append({
                                    "image_path": img_path,
                                    "caption": record.get("text", f"a photo of {self.trigger_word}")
                                })
                if len(samples) > 0:
                    return samples
            except Exception:
                pass

        # Fallback to direct directory scan pairing .png/.jpg with .txt
        img_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        for img_file in sorted(self.data_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in img_extensions:
                txt_file = img_file.with_suffix(".txt")
                caption = f"a photo of {self.trigger_word}"
                if txt_file.exists():
                    try:
                        with open(txt_file, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                            if text:
                                caption = text
                    except Exception:
                        pass
                samples.append({
                    "image_path": img_file,
                    "caption": caption
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize(self, tokenizer: Any, text: str) -> torch.Tensor:
        """Tokenizes text using HuggingFace CLIPTokenizer or fallback tensor."""
        if tokenizer is not None:
            return tokenizer(
                text,
                padding="max_length",
                max_length=self.max_token_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids[0]
        # Offline / fallback mock tokens: shape (77,)
        return torch.zeros(self.max_token_length, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        caption = sample["caption"]

        # Load and convert image to RGB
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        # Convert to tensor and normalize to [-1, 1] range expected by SDXL VAE
        # (H, W, C) uint8 -> (C, H, W) float32 in [-1, 1]
        np_img = np.array(img, dtype=np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(np_img).permute(2, 0, 1)

        # Tokenize with both SDXL tokenizers
        input_ids_1 = self._tokenize(self.tokenizer_1, caption)
        input_ids_2 = self._tokenize(self.tokenizer_2, caption)

        # SDXL micro-conditioning coordinates
        original_size = torch.tensor([orig_h, orig_w], dtype=torch.long)
        crop_coords_top_left = torch.tensor([0, 0], dtype=torch.long)
        target_size = torch.tensor([self.target_resolution, self.target_resolution], dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids_1": input_ids_1,
            "input_ids_2": input_ids_2,
            "caption": caption,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": target_size,
            "image_path": str(image_path)
        }
