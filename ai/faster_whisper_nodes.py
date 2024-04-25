# faster-whisper
# https://github.com/SYSTRAN/faster-whisper

import torchaudio
from faster_whisper import WhisperModel

from ..node_def import BASE_NODE_CATEGORY, AudioData

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
        model_input_wave = audio.waveform.clone()
        if audio.is_stereo():
            model_input_wave = model_input_wave.mean(dim=0, keepdim=True)

        WHISPER_SR = 16000
        if audio.sample_rate != WHISPER_SR:
            transform = torchaudio.transforms.Resample(
                orig_freq=audio.sample_rate, new_freq=WHISPER_SR
            )

            model_input_wave = transform(audio.waveform)

        segments, _ = model.transcribe(
            model_input_wave[0].numpy(),
            beam_size=beam_size,
            best_of=best_of,
            language=language if language != "" else None,
            initial_prompt=initial_prompt if initial_prompt != "" else None,
        )

        return (list(segments),)


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
        texts = [segment.text for segment in segments]
        return (sep.join(texts),)


class FasterWhisperListSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segments": ("FASTER_WHISPER_SEGMENTS",)}}

    CATEGORY = NODE_CATEGORY
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("FASTER_WHISPER_SEGMENT",)
    RETURN_NAMES = ("segments",)
    FUNCTION = "list"

    def list(self, segments):
        return (list(segments),)


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
