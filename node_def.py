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

    @staticmethod
    def from_comfyUI_audio(dic:dict):
        return AudioData(dic["waveform"].squeeze(0), dic["sample_rate"])

    def is_stereo(self):
        return self.waveform.size(0) > 1
    
    def to_comfyUI_audio(self):
        return { "waveform": self.waveform.unsqueeze(0), "sample_rate": self.sample_rate }


@dataclass(frozen=True)
class SpectrogramData:
    """
    spectrogram data mainly handled for visualization
    RETURN_TYPES = ("AUDIO", )
    """

    waveform: torch.Tensor

    sample_rate: int


# Node Pin
