"""
SafeTensors serialization and export engine for SDXL LoRA weights.
Saves trained LoRA adapters in the universal .safetensors format with embedded
metadata headers compatible with Diffusers, Automatic1111 WebUI, and ComfyUI.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import time
import torch
from safetensors.torch import save_file, load_file

from src.utils.config import TrainingConfig
from .lora_network import LoRANetworkManager


class LoRASaver:
    """Serializes LoRA adapters into standard .safetensors format with metadata headers."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "checkpoints",
        output_name: str = "person_sdxl_lora",
        trigger_word: str = "sks person"
    ):
        self.output_dir = Path(output_dir)
        self.output_name = output_name
        self.trigger_word = trigger_word
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_peft_to_diffusers_naming(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Translates raw PEFT weight keys into standard Diffusers/Kohya LoRA keys.
        e.g., 'base_model.model.down_blocks.0.attentions.0...lora_A.weight'
          -> 'lora_unet_down_blocks_0_attentions_0_..._lora.down.weight'
        """
        converted = {}
        for key, tensor in state_dict.items():
            clean_key = key
            # Remove wrapper prefixes
            for prefix in ("base_model.model.", "model.", "unet."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]

            # Standardize adapter naming
            if ".lora_A.weight" in clean_key:
                clean_key = clean_key.replace(".lora_A.weight", ".lora_down.weight")
            elif ".lora_B.weight" in clean_key:
                clean_key = clean_key.replace(".lora_B.weight", ".lora_up.weight")

            # Prefix with unet identifier if not present
            if not clean_key.startswith("lora_unet_") and not clean_key.startswith("lora_te_"):
                clean_key = f"lora_unet_{clean_key}"

            converted[clean_key] = tensor.contiguous() if hasattr(tensor, "contiguous") else tensor

        return converted

    def save(
        self,
        model: torch.nn.Module,
        training_config: Optional[TrainingConfig] = None,
        additional_metadata: Optional[Dict[str, str]] = None,
        step: Optional[int] = None
    ) -> Path:
        """
        Extracts LoRA weights, normalizes key names, attaches metadata headers,
        and saves to .safetensors file.
        """
        cfg = training_config or TrainingConfig()
        filename = f"{self.output_name}.safetensors"
        if step is not None:
            filename = f"{self.output_name}_step_{step:05d}.safetensors"

        safetensors_path = self.output_dir / filename

        # 1. Extract only LoRA weights
        raw_state_dict = LoRANetworkManager.extract_lora_state_dict(model)
        formatted_state_dict = self.convert_peft_to_diffusers_naming(raw_state_dict)

        # 2. Build metadata header
        metadata: Dict[str, str] = {
            "format": "pt",
            "ss_base_model_version": "sdxl_base_v1-0",
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(cfg.rank),
            "ss_network_alpha": str(cfg.lora_alpha),
            "ss_network_dropout": str(cfg.lora_dropout),
            "ss_trigger_word": self.trigger_word,
            "ss_learning_rate": str(cfg.learning_rate),
            "ss_max_train_steps": str(cfg.max_train_steps),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generator": "ai_photo_generator_phase1"
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        # 3. Save to .safetensors format
        save_file(formatted_state_dict, safetensors_path, metadata=metadata)

        # 4. Save accompanying config JSON
        manifest_path = self.output_dir / f"{safetensors_path.stem}_config.json"
        manifest_data = {
            "safetensors_file": safetensors_path.name,
            "metadata": metadata,
            "tensor_count": len(formatted_state_dict),
            "rank": cfg.rank,
            "lora_alpha": cfg.lora_alpha,
            "trigger_word": self.trigger_word,
            "total_bytes": safetensors_path.stat().st_size
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        print(f"[OK] Saved trained LoRA ({len(formatted_state_dict)} tensors) to: {safetensors_path}")
        return safetensors_path

    @staticmethod
    def inspect_safetensors(path: Union[str, Path]) -> Dict[str, Any]:
        """Reads and verifies a saved .safetensors LoRA file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"SafeTensors file not found at: {path}")

        tensors = load_file(str(p))
        return {
            "path": str(p),
            "tensor_count": len(tensors),
            "keys": list(tensors.keys())[:10],
            "file_size_bytes": p.stat().st_size
        }

