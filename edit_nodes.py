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
        audio: AudioData|dict,
        start_second: float,
        end_second: float,
    ):
        audioData = AudioData.from_comfyUI_audio(audio) if isinstance(audio,dict) else audio
        start_sample = max(0, int(start_second * audioData.sample_rate) - 1)
        end_sample = max(0, int(end_second * audioData.sample_rate) - 1)
        if start_sample == end_sample:
            return (audioData.waveform.detach().clone(),)
        view1 = audioData.waveform[..., :start_sample]
        view2 = audioData.waveform[..., end_sample:]
        return AudioData(torch.concat([view1, view2], dim=-1), audioData.sample_rate).to_comfyUI_audio()


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
        audio: AudioData|dict,
        start_second: float,
        end_second: float,
    ):
        audioData = AudioData.from_comfyUI_audio(audio) if isinstance(audio,dict) else audio
        start_sample = max(0, int(start_second * audioData.sample_rate) - 1)
        end_sample = max(0, int(end_second * audioData.sample_rate) - 1)
        if start_sample == end_sample:
            return (audioData.waveform.detach().clone(),)
        view = audioData.waveform[..., start_sample:end_sample]
        return (AudioData(view.detach().clone(), audioData.sample_rate).to_comfyUI_audio(),)


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
        audioData = AudioData.from_comfyUI_audio(audio) if isinstance(audio,dict) else audio
        if start_sample == end_sample:
            return (audioData.waveform.detach().clone(),)
        view = audioData.waveform[..., start_sample:end_sample]
        return (AudioData(view.detach().clone(), audioData.sample_rate).to_comfyUI_audio(),)


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
        audio: AudioData|dict,
        start_second: float,
        end_second: float,
    ):
        audioData = AudioData.from_comfyUI_audio(audio) if isinstance(audio,dict) else audio
        start_sample = max(0, int(start_second * audioData.sample_rate) - 1)
        end_sample = max(0, int(end_second * audioData.sample_rate) - 1)
        if start_sample == end_sample:
            return (audioData.waveform.detach().clone(),)
        copy_waveform = audioData.waveform.detach().clone()
        copy_waveform[:, start_sample:end_sample] = 0
        return (AudioData(copy_waveform, audioData.sample_rate).to_comfyUI_audio(),)


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
        return (AudioData(wave, sample_rate).to_comfyUI_audio(),)


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

    def split_audio(self, audio: AudioData|dict, second: float):
        audioData = AudioData.from_comfyUI_audio(audio) if isinstance(audio,dict) else audio
        sample = max(0, int(second * audioData.sample_rate) - 1)
        view1 = audioData.waveform[..., :sample]
        view2 = audioData.waveform[..., sample:]
        audio1 = AudioData(view1.detach().clone(), audioData.sample_rate).to_comfyUI_audio()
        audio2 = AudioData(view2.detach().clone(), audioData.sample_rate).to_comfyUI_audio()
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

    def join_audio(self, audio1: AudioData|dict, audio2: AudioData|dict, silent_interval: float):
        audioData1 = AudioData.from_comfyUI_audio(audio1) if isinstance(audio1,dict) else audio1
        audioData2 = AudioData.from_comfyUI_audio(audio2) if isinstance(audio2,dict) else audio2
        if audioData1.sample_rate != audio2.sample_rate:
            raise ValueError(
                f"sample_rate is different\naudio1: {audio1.sample_rate}\naudio2: {audio2.sample_rate}"
            )
        if not audioData1.is_stereo() == audioData2.is_stereo():
            audio1_ch = "stereo" if audioData1.is_stereo() else "monoral"
            audio2_ch = "stereo" if audioData1.is_stereo() else "monoral"
            raise ValueError(f"naudio1 is {audio1_ch} but audio2 is {audio2_ch}")

        samples = int(silent_interval * audioData1.sample_rate)
        ch_dim = 2 if audioData1.is_stereo() else 1
        interval = torch.zeros((ch_dim, samples))
        new_waveform = torch.concat(
            [audioData1.waveform, interval, audioData2.waveform], dim=-1
        )
        return (AudioData(new_waveform, audio1.sample_rate).to_comfyUI_audio(),)


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

    def join_audio(self, audios: list[AudioData|dict], silent_interval: float):
        audioDatas = map(lambda audio: AudioData.from_comfyUI_audio(audio) if isinstance(audio,dict) else audio, audios)

        if len(audioDatas) == 0:
            raise ValueError("Audios size is zero.")
        if not all([audio.sample_rate == audioDatas[0].sample_rate for audio in audioDatas]):
            raise ValueError("Sample rates must be the same.")
        if not all([audio.is_stereo() == audioDatas[0].is_stereo() for audio in audioDatas]):
            raise ValueError("All auido must be either stereo or monoral.")

        samples = int(silent_interval * audioDatas[0].sample_rate)
        ch_dim = 2 if audioDatas[0].is_stereo() else 1
        interval = torch.zeros((ch_dim, samples))
        new_tensors = []
        for audio in audioDatas:
            new_tensors.append(audio)
            new_tensors.append(interval)

        new_tensors.pop()

        new_waveforms = torch.concat(new_tensors, dim=-1)
        return (AudioData(new_waveforms, audioDatas[0].sample_rate).to_comfyUI_audio(),)


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
        audio: AudioData|dict,
        new_freq: int,
        resampling_method: str,
        lowpass_filter_width: int,
        rolloff: float,
        beta=None,
    ):
        audioData = AudioData.from_comfyUI_audio(audio) if isinstance(audio,dict) else audio
        transform = torchaudio.transforms.Resample(
            orig_freq=audioData.sample_rate,
            new_freq=new_freq,
            resampling_method=resampling_method,
            lowpass_filter_width=lowpass_filter_width,
            rolloff=rolloff,
            beta=beta,
        )
        return (AudioData(transform(audioData.waveform), new_freq).to_comfyUI_audio(),)


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
