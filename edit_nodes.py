import sys
import warnings

import torch
import torchaudio

from .node_def import BASE_NODE_CATEGORY, MAX_SAMPLE_RATE, AudioData

NODE_CATEGORY = BASE_NODE_CATEGORY + "/edit"


class CutAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_second": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "end_second": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "cut_audio"

    def cut_audio(
        self,
        audio: AudioData,
        start_second: float,
        end_second: float,
    ):
        start_sample = max(0, int(start_second * audio["sample_rate"]) - 1)
        end_sample = max(0, int(end_second * audio["sample_rate"]) - 1)
        if start_sample == end_sample:
            warnings.warn("start_sample and end_sample have the same value.")
            return (audio,)

        view1 = audio["waveform"][..., :start_sample]
        view2 = audio["waveform"][..., end_sample:]
        new_audio = {
            "waveform": torch.concat([view1, view2], dim=-1),
            "sample_rate": audio["sample_rate"],
        }
        return (new_audio,)


class TrimAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_second": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "end_second": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "trim_audio"

    def trim_audio(
        self,
        audio: AudioData,
        start_second: float,
        end_second: float,
    ):
        start_sample = max(0, int(start_second * audio["sample_rate"]) - 1)
        end_sample = max(0, int(end_second * audio["sample_rate"]) - 1)
        if start_sample == end_sample:
            warnings.warn("start_sample and end_sample have the same value.")
            return (audio,)

        view = audio["waveform"][..., start_sample:end_sample]
        new_audio = {
            "waveform": view.detach().clone(),
            "sample_rate": audio["sample_rate"],
        }
        return (new_audio,)


class TrimAudioBySample:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_sample": ("INT", {"default": 0, "min": 0, "max": 2**32}),
                "end_sample": ("INT", {"default": 0, "min": 0, "max": 2**32}),
            }
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "trim_audio"

    def trim_audio(
        self,
        audio: AudioData,
        start_sample: int,
        end_sample: int,
    ):
        if start_sample == end_sample:
            warnings.warn("start_sample and end_sample have the same value.")
            return (audio,)

        view = audio["waveform"][..., start_sample:end_sample]
        new_audio = {
            "waveform": view.detach().clone(),
            "sample_rate": audio["sample_rate"],
        }
        return (new_audio,)


class SilenceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_second": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "end_second": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "silence_audio"

    def silence_audio(
        self,
        audio: AudioData,
        start_second: float,
        end_second: float,
    ):
        start_sample = max(0, int(start_second * audio["sample_rate"]) - 1)
        end_sample = max(0, int(end_second * audio["sample_rate"]) - 1)
        if start_sample == end_sample:
            warnings.warn("start_sample and end_sample have the same value.")
            return (audio,)

        copy_waveform = audio["waveform"].detach().clone()
        copy_waveform[..., start_sample:end_sample] = 0
        new_audio = {"waveform": copy_waveform, "sample_rate": audio["sample_rate"]}
        return (new_audio,)


class MakeSilenceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "second": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "sample_rate": ("INT", {"default": 16000, "min": 1, "max": MAX_SAMPLE_RATE}),
                "channel": (["monaural", "stereo"],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "make_silence_audio"

    def make_silence_audio(
        self,
        second: float,
        sample_rate: int,
        channel: str,
        batch_size: int,
    ):
        ch_dim = 1 if channel == "monaural" else 2
        samples = int(second * sample_rate)
        wave = torch.zeros((batch_size, ch_dim, samples))
        new_audio = {"waveform": wave, "sample_rate": sample_rate}
        return (new_audio,)


class SplitAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "second": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("audio1", "audio2")
    FUNCTION = "split_audio"

    def split_audio(self, audio: AudioData, second: float):
        sample = max(0, int(second * audio["sample_rate"]) - 1)
        view1 = audio["waveform"][..., :sample]
        view2 = audio["waveform"][..., sample:]
        audio1 = {
            "waveform": view1.detach().clone(),
            "sample_rate": audio["sample_rate"],
        }
        audio2 = {
            "waveform": view2.detach().clone(),
            "sample_rate": audio["sample_rate"],
        }
        return (audio1, audio2)


class JoinAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audios": ("AUDIO",),
                "silent_interval": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    CATEGORY = NODE_CATEGORY
    INPUT_IS_LIST = True
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "join_audio"

    def join_audio(self, audios: list[AudioData], silent_interval: float):
        if len(audios) == 0:
            raise ValueError("Audios size is zero.")
        if not all(
            [audio["sample_rate"] == audios[0]["sample_rate"] for audio in audios]
        ):
            raise ValueError("sample_rate must be the same.")
        if not all(
            [
                audio["waveform"].size(1) == audios[0]["waveform"].size(1)
                for audio in audios
            ]
        ):
            raise ValueError("All audio must be either stereo or monaural.")

        samples = int(silent_interval * audios[0]["sample_rate"])
        interval = torch.zeros((audios[0]["waveform"].size(1), samples))
        new_tensors = []
        for audio in audios:
            new_tensors.append(audio)
            new_tensors.append(interval)

        new_tensors.pop()

        new_waveforms = torch.concat(new_tensors, dim=-1)
        new_audio = {"waveform": new_waveforms, "sample_rate": audios[0]["sample_rate"]}
        return (new_audio,)


class ConcatAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "silent_interval": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "concat_audio"

    def concat_audio(
        self, audio1: AudioData, audio2: AudioData, silent_interval: float
    ):
        if audio1["sample_rate"] != audio2["sample_rate"]:
            raise ValueError(
                f"sample_rate is different.\naudio1: {0}\naudio2: {1}".format(
                    audio1["sample_rate"], audio2["sample_rate"]
                )
            )

        audio1_ch = "stereo" if audio1["waveform"].size(1) > 1 else "monaural"
        audio2_ch = "stereo" if audio2["waveform"].size(1) > 1 else "monaural"
        if not audio1_ch == audio2_ch:
            raise ValueError(f"naudio1 is {audio1_ch} but audio2 is {audio2_ch}")

        samples = int(silent_interval * audio1["sample_rate"])
        interval = torch.zeros(
            (audio1["waveform"].size(0), audio1["waveform"].size(1), samples)
        )
        new_waveform = torch.concat(
            [audio1["waveform"], interval, audio2["waveform"]], dim=-1
        )
        new_audio = {"waveform": new_waveform, "sample_rate": audio1["sample_rate"]}
        return (new_audio,)


class ResampleAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "new_freq": ("INT", {"default": 32000, "min": 1, "max": MAX_SAMPLE_RATE}),
                "resampling_method": (["sinc_interp_hann", "sinc_interp_kaiser"],),
                "lowpass_filter_width": (
                    "INT",
                    {"default": 6, "min": 0, "max": 2**32},
                ),
                "rolloff": (
                    "FLOAT",
                    {
                        "default": 0.99,
                        "min": 0,
                        "max": sys.float_info.max,
                    },
                ),
            },
            "optional": {
                "beta": (
                    "FLOAT",
                    {
                        "default": 14.769656459379492,
                        "min": 0.0,
                        "max": sys.float_info.max,
                    },
                ),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "resample"

    def resample(
        self,
        audio: AudioData,
        new_freq: int,
        resampling_method: str,
        lowpass_filter_width: int,
        rolloff: float,
        beta=None,
    ):
        transform = torchaudio.transforms.Resample(
            orig_freq=audio["sample_rate"],
            new_freq=new_freq,
            resampling_method=resampling_method,
            lowpass_filter_width=lowpass_filter_width,
            rolloff=rolloff,
            beta=beta,
        )
        new_audio = {"waveform": transform(audio["waveform"]), "sample_rate": new_freq}
        return (new_audio,)


NODE_CLASS_MAPPINGS = {
    "SDT_CutAudio": CutAudio,
    "SDT_TrimAudio": TrimAudio,
    "SDT_TrimAudioBySample": TrimAudioBySample,
    "SDT_SilenceAudio": SilenceAudio,
    "SDT_MakeSilenceAudio": MakeSilenceAudio,
    "SDT_SplitAudio": SplitAudio,
    "SDT_JoinAudio": JoinAudio,
    "SDT_ConcatAudio": ConcatAudio,
    "SDT_ResampleAudio": ResampleAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_CutAudio": "Cut Audio",
    "SDT_TrimAudio": "Trim Audio",
    "SDT_TrimAudioBySample": "Trim Audio By Sample",
    "SDT_SilenceAudio": "Silence Audio",
    "SDT_MakeSilenceAudio": "Make Silence Audio",
    "SDT_SplitAudio": "Split Audio",
    "SDT_JoinAudio": "Join Audio",
    "SDT_ConcatAudio": "Concat Audio",
    "SDT_ResampleAudio": "Resample Audio",
}
