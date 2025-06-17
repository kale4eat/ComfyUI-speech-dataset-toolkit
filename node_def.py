from dataclasses import dataclass
from typing import Any, Dict

import torch

# Category

BASE_NODE_CATEGORY = "speech-dataset-toolkit"

# Value

MAX_SAFE_INT = 2**31 - 1  # 2,147,483,647
MAX_SAMPLE_RATE = 768000

# Type

AudioData = Dict[str, Any]

# Data

@dataclass(frozen=True)
class SpectrogramData:
    """
    spectrogram data mainly handled for visualization
    RETURN_TYPES = ("AUDIO", )
    """

    waveform: torch.Tensor

    sample_rate: int


# Node Pin
