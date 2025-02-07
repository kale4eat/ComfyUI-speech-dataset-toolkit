import nue_asr
import torch
import torchaudio

from ..node_def import BASE_NODE_CATEGORY, AudioData
from ..waveform_util import is_stereo, stereo_to_monaural

NODE_CATEGORY = BASE_NODE_CATEGORY + "/ai/nue-asr"


class NueAsrLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cpu", "cuda"],),
                "fp16": ("BOOLEAN", {"default": False}),
                "use_deepspeed": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("NUE_ASR",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    def load(self, device: str, fp16: bool, use_deepspeed: bool):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = nue_asr.load_model("rinna/nue-asr", device, fp16, use_deepspeed)
        tokenizer = nue_asr.load_tokenizer("rinna/nue-asr")
        return ((model, tokenizer),)


class NueAsrTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("NUE_ASR",),
                "audio": ("AUDIO",),
            }
        }

    CATEGORY = NODE_CATEGORY
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"

    def transcribe(self, model, audio: AudioData):
        model, tokenizer = model
        NUA_ASR_SR = 16000
        model_input_wave = audio["waveform"].clone()
        # input needs to be monaural
        if is_stereo(model_input_wave):
            model_input_wave = stereo_to_monaural(model_input_wave)

        if audio["sample_rate"] != NUA_ASR_SR:
            transform = torchaudio.transforms.Resample(
                orig_freq=audio["sample_rate"], new_freq=NUA_ASR_SR
            )
            model_input_wave = transform(model_input_wave)

        batch_text = []

        for b in range(model_input_wave.shape[0]):
            result = nue_asr.transcribe(model, tokenizer, model_input_wave[b])
            batch_text.append(result.text)

        return (batch_text,)


NODE_CLASS_MAPPINGS = {
    "SDT_NueAsrLoader": NueAsrLoader,
    "SDT_NueAsrTranscribe": NueAsrTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_NueAsrLoader": "Load nue-asr",
    "SDT_NueAsrTranscribe": "Transcribe by nue-asr",
}
