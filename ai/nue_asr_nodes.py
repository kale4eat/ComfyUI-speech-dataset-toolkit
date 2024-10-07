import nue_asr
import torch
import torchaudio

from ..node_def import BASE_NODE_CATEGORY, AudioData

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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"

    def transcribe(self, model, audio: AudioData|dict):
        audioData = AudioData.from_comfyUI_audio(audio) if isinstance(audio,dict) else audio
        model, tokenizer = model
        NUA_ASR_SR = 16000
        model_input_wave = audio.waveform
        if audioData.sample_rate != NUA_ASR_SR:
            transform = torchaudio.transforms.Resample(
                orig_freq=audioData.sample_rate, new_freq=NUA_ASR_SR
            )

            model_input_wave = transform(model_input_wave)

        result = nue_asr.transcribe(model, tokenizer, model_input_wave)
        return (result.text,)


NODE_CLASS_MAPPINGS = {
    "SDT_NueAsrLoader": NueAsrLoader,
    "SDT_NueAsrTranscribe": NueAsrTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_NueAsrLoader": "Load nue-asr",
    "SDT_NueAsrTranscribe": "Transcribe by nue-asr",
}
