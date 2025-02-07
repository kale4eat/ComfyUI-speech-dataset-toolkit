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
    OUTPUT_IS_LIST = (True, True, True)
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

        batch_text = []
        batch_subwords = []
        batch_segments = []

        for b in range(audio["waveform"].shape[0]):
            model_input_audio = audio_from_tensor(audio["waveform"][b], audio["sample_rate"])
            result = transcribe(model, model_input_audio)
            batch_text.append(result.text)
            batch_subwords.append(result.subwords)
            batch_segments.append(result.segments)

        return (
            batch_text, batch_subwords, batch_segments
        )


NODE_CLASS_MAPPINGS = {
    "SDT_NemoAsrLoader": NemoAsrLoader,
    "SDT_NemoAsrTranscribe": NemoAsrTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_NemoAsrLoader": "Load nemo-asr",
    "SDT_NemoAsrTranscribe": "Transcribe by nemo-asr",
}
