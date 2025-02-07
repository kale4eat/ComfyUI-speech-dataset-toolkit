# faster-whisper
# https://github.com/SYSTRAN/faster-whisper

import warnings

import torchaudio
from faster_whisper import WhisperModel

from ..node_def import BASE_NODE_CATEGORY, AudioData
from ..waveform_util import is_stereo, stereo_to_monaural

NODE_CATEGORY = BASE_NODE_CATEGORY + "/ai/faster-whisper"


class FasterWhisperLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_size": (["large-v3"],),
                "device": (["auto", "cpu", "cuda"],),
                "compute_type": (
                    [
                        "default",
                        "auto",
                        "int8",
                        "int8_float32",
                        "int8_float16",
                        "int8_bfloat16",
                        "int16",
                        "float16",
                        "float32",
                        "bfloat16",
                    ],
                ),
                "cpu_threads": ("INT", {"default": 0, "min": 0, "max": 2**10}),
                "num_workers": ("INT", {"default": 1, "min": 1, "max": 2**10}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("FASTER_WHISPER",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    def load(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        cpu_threads: int,
        num_workers: int,
    ):
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        return (model,)


class FasterWhisperTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FASTER_WHISPER",),
                "audio": ("AUDIO",),
                "beam_size": ("INT", {"default": 5, "min": 0, "max": 2**10}),
                "best_of": ("INT", {"default": 5, "min": 0, "max": 2**10}),
            },
            "optional": {
                "language": ("STRING", {"default": ""}),
                "initial_prompt": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("FASTER_WHISPER_SEGMENTS",)
    RETURN_NAMES = ("segments",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "transcribe"

    def transcribe(
        self,
        model: WhisperModel,
        audio: AudioData,
        beam_size: int,
        best_of: int,
        language: str = "",
        initial_prompt: str = "",
    ):
        model_input_wave = audio["waveform"].clone()
        # input needs to be monaural
        if is_stereo(model_input_wave):
            model_input_wave = stereo_to_monaural(model_input_wave)

        WHISPER_SR = 16000
        if audio["sample_rate"] != WHISPER_SR:
            transform = torchaudio.transforms.Resample(
                orig_freq=audio["sample_rate"], new_freq=WHISPER_SR
            )
            model_input_wave = transform(model_input_wave)

        batch_segments = []
        for b in range(model_input_wave.shape[0]):
            segments, _ = model.transcribe(
                model_input_wave[b][0].numpy(),
                beam_size=beam_size,
                best_of=best_of,
                language=language if language != "" else None,
                initial_prompt=initial_prompt if initial_prompt != "" else None,
            )

            batch_segments.append(list(segments))

        return (batch_segments,)


class FasterWhisperTextFromSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments": ("FASTER_WHISPER_SEGMENTS",),
                "sep": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_from_segments"

    def text_from_segments(self, segments, sep: str):
        texts = [s.text for s in segments]
        return (sep.join(texts),)


class FasterWhisperListSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segments": ("FASTER_WHISPER_SEGMENTS",)}}

    CATEGORY = NODE_CATEGORY
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("FASTER_WHISPER_SEGMENT",)
    RETURN_NAMES = ("segments",)
    FUNCTION = "list"

    def list(self, segments):
        if isinstance(segments, list):
            if len(segments) > 1 and isinstance(segments[0], list):
                warnings.warn("segments after batch size 2 are not processed.")

        return (list(segments[0]),)


class FasterWhisperSegmentProperty:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segment": ("FASTER_WHISPER_SEGMENT",)}}

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("start", "end", "text")
    FUNCTION = "prop"

    def prop(self, segment):
        return (segment.start, segment.end, segment.text)


NODE_CLASS_MAPPINGS = {
    "SDT_FasterWhisperLoader": FasterWhisperLoader,
    "SDT_FasterWhisperTranscribe": FasterWhisperTranscribe,
    "SDT_FasterWhisperTextFromSegments": FasterWhisperTextFromSegments,
    "SDT_FasterWhisperListSegments": FasterWhisperListSegments,
    "SDT_FasterWhisperSegmentProperty": FasterWhisperSegmentProperty,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_FasterWhisperLoader": "Load faster-whisper",
    "SDT_FasterWhisperTranscribe": "Transcribe by faster-whisper",
    "SDT_FasterWhisperTextFromSegments": "faster-whisper Text From Segments",
    "SDT_FasterWhisperListSegments": "faster-whisper List Segments",
    "SDT_FasterWhisperSegmentProperty": "faster-whisper Segment Property",
}
