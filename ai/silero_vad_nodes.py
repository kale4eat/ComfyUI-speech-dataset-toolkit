# silero-vad
# https://github.com/snakers4/silero-vad

import copy
import sys
import warnings
from typing import Callable, Optional

import torch
import torchaudio

from ..node_def import BASE_NODE_CATEGORY, MAX_SAFE_INT, AudioData
from ..waveform_util import is_stereo, stereo_to_monaural

NODE_CATEGORY = BASE_NODE_CATEGORY + "/ai/SileroVAD"

_SILERO_VAD_SR = 16000


class SileroVADLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"force_onnx_cpu": ("BOOLEAN", {"default": False})},
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("SILERO_VAD",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    def load(self, force_onnx_cpu: bool):
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=True,
            onnx=True,
            force_onnx_cpu=force_onnx_cpu,
        )
        return (model,)


# refer: https://github.com/snakers4/silero-vad/blob/master/utils_vad.py
def _get_speech_timestamps(
    audio: torch.Tensor,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 512,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    # visualize_probs: bool = False,
    progress_tracking_callback: Optional[Callable[[float], None]] = None,
):
    """
    This method is used for splitting long audios into speech chunks using silero VAD

    Parameters
    ----------
    audio: torch.Tensor, one dimensional
        One dimensional float torch.Tensor, other types are casted to torch if possible

    model: preloaded .jit silero VAD model

    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

    sampling_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 sample rates

    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out

    max_speech_duration_s: int (default -  inf)
        Maximum duration of speech chunks in seconds
        Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence that lasts more than 100ms (if any), to prevent agressive cutting.
        Otherwise, they will be split aggressively just before max_speech_duration_s.

    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it

    window_size_samples: int (default - 1536 samples)
        Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
        Values other than these may affect model perfomance!!

    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side

    return_seconds: bool (default - False)
        whether return timestamps in seconds (default - samples)

    visualize_probs: bool (default - False)
        whether draw prob hist or not

    progress_tracking_callback: Callable[[float], None] (default - None)
        callback function taking progress in percents as an argument

    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """

    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError(
                "More than one dimension in audio. Are you trying to process audio with 2 channels?"
            )

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn(
            "Sampling rate is a multiply of 16000, casting to 16000 manually!"
        )
    else:
        step = 1

    if sampling_rate == 8000 and window_size_samples > 768:
        warnings.warn(
            "window_size_samples is too big for 8000 sampling_rate! Better set window_size_samples to 256, 512 or 768 for 8000 sample rate!"
        )
    if window_size_samples not in [256, 512, 768, 1024, 1536]:
        warnings.warn(
            "Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 sampling_rate\n - [256, 512, 768] for 8000 sampling_rate"
        )

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(
                chunk, (0, int(window_size_samples - len(chunk)))
            )
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)
        # caculate progress and seng it to callback function
        progress = current_start_sample + window_size_samples
        if progress > audio_length_samples:
            progress = audio_length_samples
        progress_percent = (progress / audio_length_samples) * 100
        if progress_tracking_callback:
            progress_tracking_callback(progress_percent)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0  # to save potential segment end (and tolerate some silence)
    prev_end = next_start = (
        0  # to save potential segment limits in case of maximum segment size reached
    )

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        if (
            triggered
            and (window_size_samples * i) - current_speech["start"] > max_speech_samples
        ):
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if (
                    next_start < prev_end
                ):  # previously reached silence (< neg_thres) and is still not speech (< thres)
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech["end"] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if (
                ((window_size_samples * i) - temp_end)
                > min_silence_samples_at_max_speech
            ):  # condition to avoid cutting in very short silence
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    if return_seconds:
        for speech_dict in speeches:
            speech_dict["start"] = round(speech_dict["start"] / sampling_rate, 1)
            speech_dict["end"] = round(speech_dict["end"] / sampling_rate, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict["start"] *= step
            speech_dict["end"] *= step

    # if visualize_probs:
    #     make_visualization(speech_probs, window_size_samples / sampling_rate)

    return speeches


def _collect_chunks(tss: list[dict], wav: torch.Tensor):
    chunks = []
    for i in tss:
        chunks.append(wav[i["start"] : i["end"]])
    return torch.cat(chunks)


class SileroVADApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SILERO_VAD",),
                "audio": ("AUDIO",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0, "max": 1.0},
                ),
                "min_speech_duration_ms": (
                    "INT",
                    {"default": 250, "min": 0, "max": MAX_SAFE_INT},
                ),
                "max_speech_duration_s": (
                    "FLOAT",
                    {
                        "default": sys.float_info.max,
                        "min": 0,
                        "max": sys.float_info.max,
                    },
                ),
                "min_silence_duration_ms": (
                    "INT",
                    {"default": 100, "min": 0, "max": MAX_SAFE_INT},
                ),
                "window_size_samples": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_SAFE_INT},
                ),
                "speech_pad_ms": ("INT", {"default": 30, "min": 0, "max": MAX_SAFE_INT}),
            },
        }

    CATEGORY = NODE_CATEGORY
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("SILERO_VAD_TIMESTAMPS",)
    RETURN_NAMES = ("timestamps",)
    FUNCTION = "run"

    def run(
        self,
        model,
        audio: AudioData,
        threshold: float,
        min_speech_duration_ms: int,
        max_speech_duration_s: float,
        min_silence_duration_ms: int,
        window_size_samples: int,
        speech_pad_ms: int,
    ):
        model_input_wave = audio["waveform"].clone()
        # input needs to be monaural
        if is_stereo(model_input_wave):
            model_input_wave = stereo_to_monaural(model_input_wave)

        if audio["sample_rate"] != _SILERO_VAD_SR:
            transform = torchaudio.transforms.Resample(
                orig_freq=audio["sample_rate"], new_freq=_SILERO_VAD_SR
            )
            model_input_wave = transform(model_input_wave)

        batch_speech_timestamps = []

        for b in range(model_input_wave.shape[0]):
            speech_timestamps = _get_speech_timestamps(
                model_input_wave[b].squeeze(0),
                model,
                sampling_rate=_SILERO_VAD_SR,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                max_speech_duration_s=max_speech_duration_s,
                min_silence_duration_ms=min_silence_duration_ms,
                window_size_samples=window_size_samples,
                speech_pad_ms=speech_pad_ms,
                return_seconds=True,
            )

            batch_speech_timestamps.append(speech_timestamps)

        return (batch_speech_timestamps,)


class SileroVADListTimestamps:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"timestamps": ("SILERO_VAD_TIMESTAMPS",)}}

    CATEGORY = NODE_CATEGORY
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("SILERO_VAD_TIMESTAMP",)
    RETURN_NAMES = ("timestamp",)
    FUNCTION = "list"

    def list(self, timestamps):
        if isinstance(timestamps, list):
            if len(timestamps) > 1 and isinstance(timestamps[0], list):
                warnings.warn("timestamps after batch size 2 are not processed.")

        return (list(timestamps[0]),)


class SileroVADTimestampProperty:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"timestamp": ("SILERO_VAD_TIMESTAMP",)}}

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("start", "end")
    FUNCTION = "prop"

    def prop(self, timestamp):
        return (timestamp["start"], timestamp["end"])


class SileroVADCollectChunks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timestamps": ("SILERO_VAD_TIMESTAMPS",),
                "audio": ("AUDIO",),
            }
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "collect_chunks"

    def collect_chunks(self, timestamps, audio: AudioData):
        num_batch, num_channels, _ = audio["waveform"].shape

        if num_batch > 1:
            warnings.warn("SileroVADCollectChunks expects audio with batch size 1.")

        if len(timestamps) == 0:
            warnings.warn("VadFilter no sound")
            audio = {
                "waveform": torch.zeros((1, num_channels, 0)),
                "sample_rate": audio["sample_rate"],
            }
            return (audio,)

        # second to sample
        sample_timestamps = copy.deepcopy(timestamps)
        for item in sample_timestamps:
            item["start"] = max(0, int(item["start"] * audio["sample_rate"]) - 1)
            item["end"] = max(0, int(item["end"] * audio["sample_rate"]) - 1)

        wave1 = _collect_chunks(sample_timestamps, audio["waveform"][0][0])
        if num_channels == 2:
            wave2 = _collect_chunks(sample_timestamps, audio["waveform"][0][1])
            audio = {
                "waveform": torch.stack([wave1, wave2], dim=0).unsqueeze(0),
                "sample_rate": audio["sample_rate"],
            }
        else:
            audio = {
                "waveform": wave1.unsqueeze(0).unsqueeze(0),
                "sample_rate": audio["sample_rate"],
            }
        return (audio,)


NODE_CLASS_MAPPINGS = {
    "SDT_SileroVADLoader": SileroVADLoader,
    "SDT_SileroVADApply": SileroVADApply,
    "SDT_SileroVADListTimestamps": SileroVADListTimestamps,
    "SDT_SileroVADTimestampProperty": SileroVADTimestampProperty,
    "SDT_SileroVADCollectChunks": SileroVADCollectChunks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_SileroVADLoader": "Load Silero VAD",
    "SDT_SileroVADApply": "Apply Silero VAD",
    "SDT_SileroVADListTimestamps": "SileroVAD List Timestamps",
    "SDT_SileroVADTimestampProperty": "SileroVAD Timestamp Property",
    "SDT_SileroVADCollectChunks": "SileroVAD Collect Chunks",
}
