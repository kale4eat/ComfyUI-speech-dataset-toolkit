# demucs
# https://github.com/facebookresearch/demucs
# to use api, do
# python3 -m pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs
# requirements.txt includes torch and torchaudio
# so it may cause confusion
# use fork to avoid this
import sys

import demucs.api
import torch
from demucs.audio import convert_audio_channels

from ..node_def import BASE_NODE_CATEGORY, MAX_SAFE_INT, AudioData

NODE_CATEGORY = BASE_NODE_CATEGORY + "/ai/demcus"


class DemucsLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["htdemucs", "htdemucs_ft"],),
                "device": (["auto", "cpu", "cuda"],),
                "shifts": ("INT", {"default": 1, "min": 1, "max": MAX_SAFE_INT}),
                "overlap": (
                    "FLOAT",
                    {"default": 0.25, "min": 0, "max": sys.float_info.max},
                ),
                "split": ("BOOLEAN", {"default": True}),
                "segment": ("INT", {"default": -1, "min": -1, "max": MAX_SAFE_INT}),
                "jobs": ("INT", {"default": 0, "min": 0, "max": 2**10}),
                "progress": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("DEMUCS",)
    RETURN_NAMES = ("demucs",)
    FUNCTION = "load"

    def load(
        self,
        model: str,
        device: str,
        shifts: int,
        overlap: float,
        split: bool,
        segment: int,
        jobs: int,
        progress: bool,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        separator = demucs.api.Separator(
            model=model,
            device=device,
            shifts=shifts,
            overlap=overlap,
            split=split,
            segment=segment if segment > 0 else None,
            jobs=jobs,
            progress=progress,
        )

        return (separator,)


class DemucsApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DEMUCS",),
                "audio": ("AUDIO",),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("drums", "bass", "other", "vocals")
    FUNCTION = "run"

    def run(
        self,
        model: demucs.api.Separator,
        audio: AudioData,
    ) -> tuple[
        AudioData,
        AudioData,
        AudioData,
        AudioData,
    ]:
        # clone to avoid in-place error when different elements internally point to the same memory location
        model_input_wave = convert_audio_channels(
            audio["waveform"].clone(), model.audio_channels
        ).clone()

        num_batch, _, _ = audio["waveform"].shape

        list_drums = []
        list_bass = []
        list_other = []
        list_vocals = []

        for b in range(num_batch):
            _, separated = model.separate_tensor(
                model_input_wave[b], sr=audio["sample_rate"]
            )
            list_drums.append(separated["drums"])
            list_bass.append(separated["bass"])
            list_other.append(separated["other"])
            list_vocals.append(separated["vocals"])

        batch_drums = {
            "waveform": torch.stack(list_drums, dim=0),
            "sample_rate": audio["sample_rate"],
        }
        batch_bass = {
            "waveform": torch.stack(list_bass, dim=0),
            "sample_rate": audio["sample_rate"],
        }
        batch_other = {
            "waveform": torch.stack(list_other, dim=0),
            "sample_rate": audio["sample_rate"],
        }
        batch_vocals = {
            "waveform": torch.stack(list_vocals, dim=0),
            "sample_rate": audio["sample_rate"],
        }

        return (
            batch_drums,
            batch_bass,
            batch_other,
            batch_vocals,
        )


NODE_CLASS_MAPPINGS = {
    "SDT_DemucsLoader": DemucsLoader,
    "SDT_DemucsApply": DemucsApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_DemucsLoader": "Load Demucs",
    "SDT_DemucsApply": "Apply Demucs",
}
