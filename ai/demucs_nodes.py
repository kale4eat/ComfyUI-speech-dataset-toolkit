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

from ..node_def import BASE_NODE_CATEGORY, AudioData

NODE_CATEGORY = BASE_NODE_CATEGORY + "/ai/demcus"


class DemucsLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["htdemucs", "htdemucs_ft"],),
                "device": (["auto", "cpu", "cuda"],),
                "shifts": ("INT", {"default": 1, "min": 1, "max": 2**32}),
                "overlap": (
                    "FLOAT",
                    {"default": 0.25, "min": 0, "max": sys.float_info.max},
                ),
                "split": ("BOOLEAN", {"default": True}),
                "segment": ("INT", {"default": -1, "min": -1, "max": 2**32}),
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
        audio: dict|AudioData,
    ) -> tuple[
        dict,
        dict,
        dict,
        dict,
    ]:
        # clone to avoid in-place error when different elements internally point to the same memory location
        audioData = AudioData.from_comfyUI_audio(audio) if isinstance(audio,dict) else audio
        model_input_wave = convert_audio_channels(
            audioData.waveform.clone(), model.audio_channels
        ).clone()
        _, separated = model.separate_tensor(model_input_wave, sr=audioData.sample_rate)
        return (
            AudioData(separated["drums"], audioData.sample_rate).to_comfyUI_audio(),
            AudioData(separated["bass"], audioData.sample_rate).to_comfyUI_audio(),
            AudioData(separated["other"], audioData.sample_rate).to_comfyUI_audio(),
            AudioData(separated["vocals"], audioData.sample_rate).to_comfyUI_audio(),
        )


NODE_CLASS_MAPPINGS = {
    "SDT_DemucsLoader": DemucsLoader,
    "SDT_DemucsApply": DemucsApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_DemucsLoader": "Load Demucs",
    "SDT_DemucsApply": "Apply Demucs",
}
