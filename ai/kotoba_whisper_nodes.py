# kotoba-whisper
# https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0
import re
import warnings

import torch
import torchaudio
from transformers import Pipeline, pipeline

from ..node_def import BASE_NODE_CATEGORY, AudioData
from ..waveform_util import is_stereo, stereo_to_monaural

NODE_CATEGORY = BASE_NODE_CATEGORY + "/ai/kotoba-whisper"

# TODO: support kotoba-whisper-v2.1
# TODO: support kotoba-whisper-v2.2


class KotobaWhisperLoaderShort:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (
                    [
                        "kotoba-tech/kotoba-whisper-v1.0",
                        "kotoba-tech/kotoba-whisper-v2.0",
                    ],
                ),
                "device": (["auto", "cpu", "cuda"],),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("KOTOBA_WHISPER_SHORT",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    def load(self, model_id: str, device: str):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        model_kwargs = {"attn_implementation": "sdpa"} if device == "cuda" else {}

        # load model
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs,
        )
        return (pipe,)


class KotobaWhisperLoaderLong:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (
                    [
                        "kotoba-tech/kotoba-whisper-v1.0",
                        "kotoba-tech/kotoba-whisper-v2.0",
                    ],
                ),
                "device": (["auto", "cpu", "cuda"],),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 2**10}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("KOTOBA_WHISPER_LONG",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    def load(
        self,
        model_id: str,
        device: str,
        batch_size: int,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        model_kwargs = {"attn_implementation": "sdpa"} if device == "cuda" else {}

        # load model
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs,
            batch_size=batch_size,
        )
        return (pipe,)


class KotobaWhisperTranscribeShort:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("KOTOBA_WHISPER_SHORT",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "prompt": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = NODE_CATEGORY
    OUTPUT_IS_LIST = (True, True)
    RETURN_TYPES = ("STRING", "KOTOBA_WHISPER_SEGMENTS")
    RETURN_NAMES = ("text", "segments")
    FUNCTION = "transcribe"

    def transcribe(
        self,
        model: Pipeline,
        audio: AudioData,
        prompt: str = "",
    ):
        pipe = model
        model_input_wave = audio["waveform"].clone()
        # input needs to be monaural
        if is_stereo(model_input_wave):
            model_input_wave = stereo_to_monaural(model_input_wave)

        WHISPER_SR = 16000
        if audio["sample_rate"] != WHISPER_SR:
            transform = torchaudio.transforms.Resample(
                orig_freq=audio["sample_rate"], new_freq=WHISPER_SR
            )
            model_input_wave = transform(audio["waveform"])

        batch_text = []
        batch_chunks = []

        generate_kwargs = {"language": "japanese", "task": "transcribe"}
        if len(prompt):
            generate_kwargs["prompt_ids"] = pipe.tokenizer.get_prompt_ids(  # type: ignore
                prompt, return_tensors="pt"
            ).to(pipe.device)

        for b in range(model_input_wave.shape[0]):
            model_input_wave = model_input_wave[b].numpy()[0]

            result = pipe(
                model_input_wave,
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
            )
            text = result["text"]  # type: ignore

            if len(prompt):
                # currently the pipeline for ASR appends the prompt at the beginning of the transcription, so remove it
                # refer: https://hamaruki.com/introduction-to-kotoba-whisper-a-new-option-for-japanese-speech-recognition/
                text = re.sub(rf"\A\s*{prompt}\s*", "", text)  # type: ignore

            batch_text.append(text)
            batch_chunks.append(result["chunks"])  # type: ignore

        return (batch_text, batch_chunks)


class KotobaWhisperTranscribeLong:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("KOTOBA_WHISPER_LONG",),
                "audio": ("AUDIO",),
                "chunk_length_s": ("INT", {"default": 15, "min": 1, "max": 2**10}),
            },
        }

    CATEGORY = NODE_CATEGORY
    OUTPUT_IS_LIST = (True, True)
    RETURN_TYPES = ("STRING", "KOTOBA_WHISPER_SEGMENTS")
    RETURN_NAMES = ("text", "segments")
    FUNCTION = "transcribe"

    def transcribe(
        self,
        model: Pipeline,
        audio: AudioData,
        chunk_length_s: int,
    ):
        pipe = model
        model_input_wave = audio["waveform"].clone()

        # input needs to be monaural
        if is_stereo(model_input_wave):
            model_input_wave = stereo_to_monaural(model_input_wave)

        WHISPER_SR = 16000
        if audio["sample_rate"] != WHISPER_SR:
            transform = torchaudio.transforms.Resample(
                orig_freq=audio["sample_rate"], new_freq=WHISPER_SR
            )
            model_input_wave = transform(audio["waveform"])

        batch_text = []
        batch_chunks = []

        generate_kwargs = {"language": "japanese", "task": "transcribe"}

        for b in range(model_input_wave.shape[0]):
            model_input_wave = model_input_wave[b].numpy()[0]

            result = pipe(
                model_input_wave,
                chunk_length_s=chunk_length_s,
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
            )

            batch_text.append(result["text"])  # type: ignore
            batch_chunks.append(result["chunks"])  # type: ignore

        return (batch_text, batch_chunks)


class KotobaWhisperListSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segments": ("KOTOBA_WHISPER_SEGMENTS",)}}

    CATEGORY = NODE_CATEGORY
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("KOTOBA_WHISPER_SEGMENT",)
    RETURN_NAMES = ("segments",)
    FUNCTION = "list"

    def list(self, segments):
        if isinstance(segments, list):
            if len(segments) > 1 and isinstance(segments[0], list):
                warnings.warn("segments after batch size 2 are not processed.")

        return (list(segments[0]),)


class KotobaWhisperSegmentProperty:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segment": ("KOTOBA_WHISPER_SEGMENT",)}}

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("start", "end", "text")
    FUNCTION = "prop"

    def prop(self, segment):
        return (segment["timestamp"][0], segment["timestamp"][1], segment["text"])


NODE_CLASS_MAPPINGS = {
    "SDT_KotobaWhisperLoaderShort": KotobaWhisperLoaderShort,
    "SDT_KotobaWhisperLoaderLong": KotobaWhisperLoaderLong,
    "SDT_KotobaWhisperTranscribeShort": KotobaWhisperTranscribeShort,
    "SDT_KotobaWhisperTranscribeLong": KotobaWhisperTranscribeLong,
    "SDT_KotobaWhisperListSegments": KotobaWhisperListSegments,
    "SDT_KotobaWhisperSegmentProperty": KotobaWhisperSegmentProperty,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_KotobaWhisperLoaderShort": "Load kotoba-whisper (Short-Form)",
    "SDT_KotobaWhisperLoaderLong": "Load kotoba-whisper (Long-Form)",
    "SDT_KotobaWhisperTranscribeShort": "Transcribe by kotoba-whisper (Short-Form)",
    "SDT_KotobaWhisperTranscribeLong": "Transcribe by kotoba-whisper (Long-Form)",
    "SDT_KotobaWhisperListSegments": "kotoba-whisper List Segments",
    "SDT_KotobaWhisperSegmentProperty": "kotoba-whisper Segment Property",
}
