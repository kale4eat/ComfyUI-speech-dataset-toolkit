import sys

import torch
import torchaudio

from .node_def import BASE_NODE_CATEGORY, AudioData

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
        start_sample = max(0, int(start_second * audio.sample_rate) - 1)
        end_sample = max(0, int(end_second * audio.sample_rate) - 1)
        if start_sample == end_sample:
            return (audio.waveform.detach().clone(),)
        view1 = audio.waveform[..., :start_sample]
        view2 = audio.waveform[..., end_sample:]
        return AudioData(torch.concat([view1, view2], dim=-1), audio.sample_rate)


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
        start_sample = max(0, int(start_second * audio.sample_rate) - 1)
        end_sample = max(0, int(end_second * audio.sample_rate) - 1)
        if start_sample == end_sample:
            return (audio.waveform.detach().clone(),)
        view = audio.waveform[..., start_sample:end_sample]
        return (AudioData(view.detach().clone(), audio.sample_rate),)


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
        start_sample = max(0, int(start_second * audio.sample_rate) - 1)
        end_sample = max(0, int(end_second * audio.sample_rate) - 1)
        if start_sample == end_sample:
            return (audio.waveform.detach().clone(),)
        copy_waveform = audio.waveform.detach().clone()
        copy_waveform[:, start_sample:end_sample] = 0
        return (AudioData(copy_waveform, audio.sample_rate),)


class MakeSilenceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "second": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "sample_rate": ("INT", {"default": 16000, "min": 1}),
                "channel": (["monoral", "stereo"],),
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
    ):
        ch_dim = 1 if channel == "monoral" else 2
        samples = int(second * sample_rate)
        wave = torch.zeros((ch_dim, samples))
        return (AudioData(wave, sample_rate),)


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

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "split_audio"

    def split_audio(self, audio: AudioData, second: float):
        sample = max(0, int(second * audio.sample_rate) - 1)
        view1 = audio.waveform[..., :sample]
        view2 = audio.waveform[..., sample:]
        audio1 = AudioData(view1.detach().clone(), audio.sample_rate)
        audio2 = AudioData(view2.detach().clone(), audio.sample_rate)
        return (audio1, audio2)


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
    FUNCTION = "join_audio"

    def join_audio(self, audio1: AudioData, audio2: AudioData, silent_interval: float):
        if audio1.sample_rate != audio2.sample_rate:
            raise ValueError(
                f"sample_rate is different\naudio1: {audio1.sample_rate}\naudio2: {audio2.sample_rate}"
            )
        if not audio1.is_stereo() == audio2.is_stereo():
            audio1_ch = "stereo" if audio1.is_stereo() else "monoral"
            audio2_ch = "stereo" if audio1.is_stereo() else "monoral"
            raise ValueError(f"naudio1 is {audio1_ch} but audio2 is {audio2_ch}")

        samples = int(silent_interval * audio1.sample_rate)
        ch_dim = 2 if audio1.is_stereo() else 1
        interval = torch.zeros((ch_dim, samples))
        new_waveform = torch.concat(
            [audio1.waveform, interval, audio2.waveform], dim=-1
        )
        return (AudioData(new_waveform, audio1.sample_rate),)


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
        if not all([audio.sample_rate == audios[0].sample_rate for audio in audios]):
            raise ValueError("Sample rates must be the same.")
        if not all([audio.is_stereo() == audios[0].is_stereo() for audio in audios]):
            raise ValueError("All auido must be either stereo or monoral.")

        samples = int(silent_interval * audios[0].sample_rate)
        ch_dim = 2 if audios[0].is_stereo() else 1
        interval = torch.zeros((ch_dim, samples))
        new_tensors = []
        for audio in audios:
            new_tensors.append(audio)
            new_tensors.append(interval)

        new_tensors.pop()

        new_waveforms = torch.concat(new_tensors, dim=-1)
        return (AudioData(new_waveforms, audios[0].sample_rate),)


class ResampleAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "new_freq": ("INT", {"default": 32000, "min": 1}),
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
            orig_freq=audio.sample_rate,
            new_freq=new_freq,
            resampling_method=resampling_method,
            lowpass_filter_width=lowpass_filter_width,
            rolloff=rolloff,
            beta=beta,
        )
        return (AudioData(transform(audio.waveform), new_freq),)


NODE_CLASS_MAPPINGS = {
    "SDT_CutAudio": CutAudio,
    "SDT_TrimAudio": TrimAudio,
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
    "SDT_SilenceAudio": "Silence Audio",
    "SDT_MakeSilenceAudio": "Make Silence Audio",
    "SDT_SplitAudio": "Split Audio",
    "SDT_JoinAudio": "Join Audio",
    "SDT_ConcatAudio": "Concat Audio",
    "SDT_ResampleAudio": "Resample Audio",
}
