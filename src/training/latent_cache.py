"""
Latent and Text Embedding caching system for SDXL LoRA fine-tuning.
Pre-encodes images via VAE and captions via dual CLIP text encoders to disk/memory,
allowing 3-5x faster training loops and drastically reduced VRAM footprint.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class LatentCache:
    """Manages pre-computation and retrieval of cached VAE latents and text embeddings."""

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/latents_cache",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        self.cache_dir = Path(cache_dir)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_dataset(
        self,
        dataset: Dataset,
        vae: Optional[Any] = None,
        text_encoder_1: Optional[Any] = None,
        text_encoder_2: Optional[Any] = None,
        batch_size: int = 1
    ) -> List[Path]:
        """
        Pre-computes and saves latents and prompt embeddings for every item in dataset.
        Returns list of cached file paths.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        cached_paths: List[Path] = []

        print(f"[INFO] Caching latents and text embeddings to '{self.cache_dir}'...")

        for idx, batch in enumerate(tqdm(loader, desc="Caching latents")):
            cache_file = self.cache_dir / f"latent_{idx:04d}.pt"

            with torch.no_grad():
                # 1. Compute VAE Latent
                if vae is not None:
                    pixel_values = batch["pixel_values"].to(self.device, dtype=self.dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    # Scale latents according to SDXL VAE scaling factor
                    scaling_factor = getattr(vae.config, "scaling_factor", 0.13025)
                    latents = latents * scaling_factor
                else:
                    # Mock/synthetic latent: SDXL 1024x1024 produces 128x128 latent
                    b = batch["pixel_values"].shape[0]
                    latents = torch.randn((b, 4, 128, 128), dtype=self.dtype)

                # 2. Compute Dual Text Embeddings
                prompt_embeds = None
                pooled_prompt_embeds = None

                if text_encoder_1 is not None and text_encoder_2 is not None:
                    input_ids_1 = batch["input_ids_1"].to(self.device)
                    input_ids_2 = batch["input_ids_2"].to(self.device)

                    res_1 = text_encoder_1(input_ids_1, output_hidden_states=True)
                    res_2 = text_encoder_2(input_ids_2, output_hidden_states=True)

                    # Penultimate hidden states
                    hidden_1 = res_1.hidden_states[-2]  # (B, 77, 768)
                    hidden_2 = res_2.hidden_states[-2]  # (B, 77, 1280)
                    prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)  # (B, 77, 2048)
                    pooled_prompt_embeds = res_2[0]  # (B, 1280)
                else:
                    # Mock text embeddings for SDXL
                    b = batch["pixel_values"].shape[0]
                    prompt_embeds = torch.randn((b, 77, 2048), dtype=self.dtype)
                    pooled_prompt_embeds = torch.randn((b, 1280), dtype=self.dtype)

                cache_payload = {
                    "latents": latents.cpu(),
                    "prompt_embeds": prompt_embeds.cpu(),
                    "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
                    "original_size": batch["original_size"].cpu(),
                    "crop_coords_top_left": batch["crop_coords_top_left"].cpu(),
                    "target_size": batch["target_size"].cpu(),
                    "caption": batch["caption"]
                }
                torch.save(cache_payload, cache_file)
                cached_paths.append(cache_file)

        return cached_paths


class CachedLatentDataset(Dataset):
    """Dataset that loads pre-computed VAE latents and prompt embeddings directly from disk."""

    def __init__(self, cache_dir: Union[str, Path] = "data/latents_cache"):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(list(self.cache_dir.glob("latent_*.pt")))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = torch.load(self.files[idx], weights_only=False)
        # Squeeze batch dimension if cached with batch_size=1
        return {
            "latents": data["latents"].squeeze(0),
            "prompt_embeds": data["prompt_embeds"].squeeze(0),
            "pooled_prompt_embeds": data["pooled_prompt_embeds"].squeeze(0),
            "original_size": data["original_size"].squeeze(0),
            "crop_coords_top_left": data["crop_coords_top_left"].squeeze(0),
            "target_size": data["target_size"].squeeze(0),
            "caption": data["caption"][0] if isinstance(data["caption"], list) else data["caption"]
        }
