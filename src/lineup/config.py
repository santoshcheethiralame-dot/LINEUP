from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("LINEUP_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("LINEUP_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SEED = 0


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = 256
    seed: int = DEFAULT_SEED
    dtype: str = "bfloat16"      # applied on CUDA; CPU runs fall back to float32
    device: str | None = None    # None selects cuda when available, else cpu
    load_in_4bit: bool = False   # 4-bit (nf4) so the 7B model fits a 16 GB GPU


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
