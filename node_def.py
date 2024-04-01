from dataclasses import dataclass

import torch

# Category

BASE_NODE_CATEGORY = "speech-dataset-toolkit"


# Data


@dataclass(frozen=True)
class AudioData:
    """
    audio data mainly handled in this extension
    RETURN_TYPES = ("AUDIO", )
    """

    waveform: torch.Tensor

    sample_rate: int

    def is_stereo(self):
        return self.waveform.size(0) > 1


@dataclass(frozen=True)
class SpectrogramData:
    """
    spectrogram data mainly handled for visualization
    RETURN_TYPES = ("AUDIO", )
    """

    waveform: torch.Tensor

    sample_rate: int


# Node Pin
