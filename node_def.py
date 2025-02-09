from dataclasses import dataclass
from typing import Any, Dict

import torch

# Category

BASE_NODE_CATEGORY = "speech-dataset-toolkit"

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
