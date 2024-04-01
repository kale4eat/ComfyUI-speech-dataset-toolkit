import io

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.functional as F
from PIL import Image

from .node_def import BASE_NODE_CATEGORY, AudioData, SpectrogramData

NODE_CATEGORY = BASE_NODE_CATEGORY + "/visualize"


# PIL to Tensor
# refer: https://github.com/a1lazydog/ComfyUI-AudioScheduler/blob/main/nodes.py
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class PlotWaveForm:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "title": ("STRING", {"default": "Waveform"}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph_image",)
    FUNCTION = "plot_waveform"

    def plot_waveform(self, audio: AudioData, title: str):
        # refer: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
        waveform = audio.waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / audio.sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle(title)

        # Create an in-memory buffer to store the image
        buffer = io.BytesIO()

        # Save the plot to the in-memory buffer as a PNG
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Create a Pillow Image object
        image = Image.open(buffer)

        return (pil2tensor(image),)


class PlotSpecgram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "title": ("STRING", {"default": "Spectrogram"}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph_image",)
    FUNCTION = "plot_specgram"

    def plot_specgram(self, audio: AudioData, title: str):
        # refer: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
        waveform = audio.waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / audio.sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=audio.sample_rate)
        figure.suptitle(title)

        # Create an in-memory buffer to store the image
        buffer = io.BytesIO()

        # Save the plot to the in-memory buffer as a PNG
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Create a Pillow Image object
        image = Image.open(buffer)

        return (pil2tensor(image),)


class PlotSpectrogram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spec": ("SPEC",),
                "title": ("STRING", {"default": "Spectrogram (db)"}),
                "ylabel": ("STRING", {"default": "freq_bin"}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph_image",)
    FUNCTION = "plot_spectrogram"

    def plot_spectrogram(self, spec: SpectrogramData, title: str, ylabel: str):
        # refer: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title)
        axs.set_ylabel(ylabel)
        axs.set_xlabel("frame")
        im = axs.imshow(
            librosa.power_to_db(spec.waveform[0].numpy()), origin="lower", aspect="auto"
        )
        fig.colorbar(im, ax=axs)

        # Create an in-memory buffer to store the image
        buffer = io.BytesIO()

        # Save the plot to the in-memory buffer as a PNG
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Create a Pillow Image object
        image = Image.open(buffer)

        return (pil2tensor(image),)


class PlotMelFilterBank:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "n_fft": ("INT", {"default": 256, "min": 0, "max": 2**32}),
                "n_mels": ("INT", {"default": 64, "min": 0, "max": 2**32}),
                "sample_rate": ("INT", {"default": 32000, "min": 0, "max": 2**32}),
                "title": ("STRING", {"default": "Filter bank"}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph_image",)
    FUNCTION = "plot_mel_fbank"

    def plot_mel_fbank(self, n_fft, n_mels, sample_rate, title: str):
        fbank = F.melscale_fbanks(
            int(n_fft // 2 + 1),
            n_mels=n_mels,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            sample_rate=sample_rate,
            norm="slaney",
        )
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title)
        axs.imshow(fbank, aspect="auto")
        axs.set_ylabel("frequency bin")
        axs.set_xlabel("mel bin")
        # Create an in-memory buffer to store the image
        buffer = io.BytesIO()

        # Save the plot to the in-memory buffer as a PNG
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Create a Pillow Image object
        image = Image.open(buffer)

        return (pil2tensor(image),)


class PlotPitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph_image",)
    FUNCTION = "plot_pitch"

    def plot_pitch(self, audio: AudioData):
        pitch = F.detect_pitch_frequency(audio.waveform, audio.sample_rate)
        figure, axis = plt.subplots(1, 1)
        axis.set_title("Pitch Feature")
        axis.grid(True)

        end_time = audio.waveform.shape[1] / audio.sample_rate
        time_axis = torch.linspace(0, end_time, audio.waveform.shape[1])
        axis.plot(time_axis, audio.waveform[0], linewidth=1, color="gray", alpha=0.3)

        axis2 = axis.twinx()
        time_axis = torch.linspace(0, end_time, pitch.shape[1])
        ln2 = axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

        axis2.legend(loc=0)

        # Create an in-memory buffer to store the image
        buffer = io.BytesIO()

        # Save the plot to the in-memory buffer as a PNG
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Create a Pillow Image object
        image = Image.open(buffer)

        return (pil2tensor(image),)


NODE_CLASS_MAPPINGS = {
    "SDT_PlotWaveForm": PlotWaveForm,
    "SDT_PlotSpecgram": PlotSpecgram,
    "SDT_PlotSpectrogram": PlotSpectrogram,
    "SDT_PlotMelFilterBank": PlotMelFilterBank,
    "SDT_PlotPitch": PlotPitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_PlotWaveForm": "Plot WaveForm",
    "SDT_PlotSpecgram": "Plot Specgram",
    "SDT_PlotSpectrogram": "Plot Spectrogram",
    "SDT_PlotMelFilterBank": "Plot MelFilterBank",
    "SDT_PlotPitch": "Plot Pitch",
}
