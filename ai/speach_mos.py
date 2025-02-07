# SpeechMOS
# https://github.com/tarepan/SpeechMOS


import torch

from ..node_def import BASE_NODE_CATEGORY, AudioData
from ..waveform_util import is_stereo, stereo_to_monaural

NODE_CATEGORY = BASE_NODE_CATEGORY + "/ai/SpeechMOS"


class SpeechMOSLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"device": (["auto", "cpu", "cuda"],)},
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("SPEECH_MOS",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    def load(self, device: str):
        model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
        if device == "auto" and torch.cuda.is_available():
            model = model.to("cuda")
        elif device == "cuda":
            model = model.to("cuda")
        return (model,)


class SpeechMOSScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SPEECH_MOS",),
                "audio": ("AUDIO",),
            }
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "score"

    def score(
        self,
        model,
        audio: AudioData,
    ):
        num_batch, _, _ = audio["waveform"].shape
        model_input_wave = audio["waveform"].clone()
        # input needs to be monaural
        if is_stereo(model_input_wave):
            model_input_wave = stereo_to_monaural(model_input_wave)

        device = next(model.parameters()).device
        scores = []
        with torch.no_grad():
            for b in range(num_batch):
                wave = model_input_wave[b].clone().to(device)
                score = model(wave, audio["sample_rate"]).item()
                scores.append(score)
                del wave
                torch.cuda.empty_cache()

        return (scores,)


NODE_CLASS_MAPPINGS = {
    "SDT_SpeechMOSLoader": SpeechMOSLoader,
    "SDT_SpeechMOSScore": SpeechMOSScore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_SpeechMOSLoader": "Load SpeechMOS",
    "SDT_SpeechMOSScore": "SpeechMOS Score",
}
