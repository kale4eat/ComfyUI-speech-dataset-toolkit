# kotoba-whisper
# https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0
import re

import torch
import torchaudio
from transformers import Pipeline, pipeline

from ..node_def import BASE_NODE_CATEGORY, AudioData

NODE_CATEGORY = BASE_NODE_CATEGORY + "/ai/kotoba-whisper"


class KotobaWhisperLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cpu", "cuda"],),
                "form_length": (["short", "long"],),
                "chunk_length_s": ("INT", {"default": 15, "min": 1, "max": 2**10}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 2**10}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("KOTOBA_WHISPER",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    def load(
        self,
        device: str,
        form_length: str,
        chunk_length_s: int,
        batch_size: int,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        model_kwargs = {"attn_implementation": "sdpa"} if device == "cuda" else {}
        model_id = "kotoba-tech/kotoba-whisper-v1.0"

        # load model
        if form_length == "short":
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch_dtype,
                device=device,
                model_kwargs=model_kwargs,
            )
        else:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch_dtype,
                device=device,
                model_kwargs=model_kwargs,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
            )
        return (pipe,)


class KotobaWhisperTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("KOTOBA_WHISPER",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "prompt": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"

    def transcribe(
        self,
        model: Pipeline,
        audio: AudioData,
        prompt: str = "",
    ):
        pipe = model
        model_input_wave = audio.waveform.clone()
        if audio.is_stereo():
            model_input_wave = model_input_wave.mean(dim=0, keepdim=True)

        WHISPER_SR = 16000
        if audio.sample_rate != WHISPER_SR:
            transform = torchaudio.transforms.Resample(
                orig_freq=audio.sample_rate, new_freq=WHISPER_SR
            )

            model_input_wave = transform(audio.waveform)

        model_input_wave = model_input_wave.numpy()[0]
        generate_kwargs = {"language": "japanese", "task": "transcribe"}
        if prompt == "":
            result = pipe(model_input_wave, generate_kwargs=generate_kwargs)
            return (result["text"],)  # type: ignore

        generate_kwargs["prompt_ids"] = pipe.tokenizer.get_prompt_ids(  # type: ignore
            prompt, return_tensors="pt"
        ).to(pipe.device)
        result = pipe(model_input_wave, generate_kwargs=generate_kwargs)
        text = result["text"]  # type: ignore
        # currently the pipeline for ASR appends the prompt at the beginning of the transcription, so remove it
        text = re.sub(rf"\A\s*{prompt}\s*", "", text)  # type: ignore
        return (text,)


NODE_CLASS_MAPPINGS = {
    "SDT_KotobaWhisperLoader": KotobaWhisperLoader,
    "SDT_KotobaWhisperTranscribe": KotobaWhisperTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_KotobaWhisperLoader": "Load kotoba-whisper",
    "SDT_KotobaWhisperTranscribe": "Transcribe by kotoba-whisper",
}
