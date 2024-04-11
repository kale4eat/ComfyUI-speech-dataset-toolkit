# ReazonSpeech nemo-asr
# https://github.com/reazon-research/ReazonSpeech/tree/master/pkg/nemo-asr
import torch
from reazonspeech.nemo.asr import audio_from_tensor, load_model, transcribe

from ..node_def import BASE_NODE_CATEGORY, AudioData

NODE_CATEGORY = BASE_NODE_CATEGORY + "/ai/nemo-asr"


class NemoAsrLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"device": (["auto", "cpu", "cuda"],)},
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("NEMO_ASR",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    def load(self, device):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(device)
        return (model,)


class NemoAsrTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("NEMO_ASR",),
                "audio": ("AUDIO",),
            }
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = (
        "STRING",
        "NEMO_ASR_SUBWORDS",
        "NEMO_ASR_SEGMENTS",
    )
    RETURN_NAMES = (
        "text",
        "subwords",
        "segments",
    )

    FUNCTION = "transcribe"

    def transcribe(
        self,
        model,
        audio: AudioData,
    ):
        model_input_audio = audio_from_tensor(audio.waveform, audio.sample_rate)
        result = transcribe(model, model_input_audio)
        return (
            result.text,
            result.subwords,
            result.segments,
        )


class NemoAsrListSubwords:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"subwords": ("NEMO_ASR_SUBWORDS",)}}

    CATEGORY = NODE_CATEGORY
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("NEMO_ASR_SUBWORD",)
    RETURN_NAMES = ("subwords",)
    FUNCTION = "list"

    def list(self, segment):
        return (list(segment),)


class NemoAsrSubwordProperty:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"subword": ("NEMO_ASR_SUBWORD",)}}

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("seconds", "token_id", "token")
    FUNCTION = "prop"

    def prop(self, subword):
        return (subword.seconds, subword.token_id, subword.token)


class NemoAsrListSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segments": ("NEMO_ASR_SEGMENTS",)}}

    CATEGORY = NODE_CATEGORY
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("NEMO_ASR_SEGMENT",)
    RETURN_NAMES = ("segments",)
    FUNCTION = "list"

    def list(self, segments):
        return (segments,)


class NemoAsrSegmentProperty:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segment": ("NEMO_ASR_SEGMENT",)}}

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("start", "end", "text")
    FUNCTION = "prop"

    def prop(self, segment):
        return (segment.start_seconds, segment.end_seconds, segment.text)


NODE_CLASS_MAPPINGS = {
    "SDT_NemoAsrLoader": NemoAsrLoader,
    "SDT_NemoAsrTranscribe": NemoAsrTranscribe,
    "SDT_NemoAsrListSubwords": NemoAsrListSubwords,
    "SDT_NemoAsrSubwordProperty": NemoAsrSubwordProperty,
    "SDT_NemoAsrListSegments": NemoAsrListSegments,
    "SDT_NemoAsrSegmentProperty": NemoAsrSegmentProperty,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_NemoAsrLoader": "Load nemo-asr",
    "SDT_NemoAsrTranscribe": "Transcribe by nemo-asr",
    "SDT_NemoAsrListSubwords": "nemo-asr List Subwords",
    "SDT_NemoAsrSubwordProperty": "nemo-asr Subword Property",
    "SDT_NemoAsrListSegments": "nemo-asr List Segments",
    "SDT_NemoAsrSegmentProperty": "nemo-asr Segment Property",
}
