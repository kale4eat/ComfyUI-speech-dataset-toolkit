import sys

import torchaudio.functional as F

from .node_def import BASE_NODE_CATEGORY, AudioData

NODE_CATEGORY = BASE_NODE_CATEGORY + "/filter"


class HighpassBiquad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "cutoff_freq": (
                    "FLOAT",
                    {"default": 0, "min": 0, "max": sys.float_info.max},
                ),
                "Q": (
                    "FLOAT",
                    {"default": 0.707, "min": 0, "max": sys.float_info.max},
                ),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "highpass_biquad"

    def highpass_biquad(self, audio: AudioData, cutoff_freq: float, Q: float):
        waveform = F.highpass_biquad(
            audio["waveform"], audio["sample_rate"], cutoff_freq, Q
        )
        new_audio = {"waveform": waveform, "sample_rate": audio["sample_rate"]}
        return (new_audio,)


class LowpassBiquad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "cutoff_freq": (
                    "FLOAT",
                    {"default": 0, "min": 0, "max": sys.float_info.max},
                ),
                "Q": ("FLOAT", {"default": 0.707, "min": 0, "max": sys.float_info.max}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "lowpass_biquad"

    def lowpass_biquad(self, audio: AudioData, cutoff_freq: float, Q: float):
        waveform = F.lowpass_biquad(audio["waveform"], audio["sample_rate"], cutoff_freq, Q)
        new_audio = {"waveform": waveform, "sample_rate": audio["sample_rate"]}
        return (new_audio,)


NODE_CLASS_MAPPINGS = {
    "SDT_HighpassBiquad": HighpassBiquad,
    "SDT_LowpassBiquad": LowpassBiquad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_HighpassBiquad": "Highpass Biquad",
    "SDT_LowpassBiquad": "Lowpass Biquad",
}
