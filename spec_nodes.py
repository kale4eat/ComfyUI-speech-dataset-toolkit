import torchaudio.transforms as T

from .node_def import BASE_NODE_CATEGORY, MAX_SAFE_INT, AudioData, SpectrogramData

NODE_CATEGORY = BASE_NODE_CATEGORY + "/spec"


class Spectrogram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "n_fft": ("INT", {"default": 400, "min": 0, "max": MAX_SAFE_INT}),
            },
            "optional": {
                "win_length": ("INT", {"default": -1, "min": -1, "max": MAX_SAFE_INT}),
                "hop_length": ("INT", {"default": -1, "min": -1, "max": MAX_SAFE_INT}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("SPEC",)
    RETURN_NAMES = ("spec",)
    FUNCTION = "spectrogram"

    def spectrogram(self, audio: AudioData, n_fft, win_length=None, hop_length=None):
        spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length if win_length != -1 else None,
            hop_length=hop_length if hop_length != -1 else None,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )

        spec = SpectrogramData(spectrogram(audio["waveform"]), audio["sample_rate"])
        return (spec,)


class GriffinLim:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spec": ("SPEC",),
                "n_fft": ("INT", {"default": 400, "min": 0, "max": MAX_SAFE_INT}),
            },
            "optional": {
                "win_length": ("INT", {"default": -1, "min": -1, "max": MAX_SAFE_INT}),
                "hop_length": ("INT", {"default": -1, "min": -1, "max": MAX_SAFE_INT}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "griffin_lim"

    def griffin_lim(
        self, spec: SpectrogramData, n_fft, win_length=None, hop_length=None
    ):
        griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            win_length=win_length if win_length != -1 else None,
            hop_length=hop_length if hop_length != -1 else None,
        )

        waveform = griffin_lim(spec.waveform)
        audio = {"waveform": waveform, "sample_rate": spec.sample_rate}
        return (audio,)


class MelSpectrogram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "n_fft": ("INT", {"default": 400, "min": 0, "max": MAX_SAFE_INT}),
                "n_mels": ("INT", {"default": 128, "min": 0, "max": MAX_SAFE_INT}),
            },
            "optional": {
                "win_length": ("INT", {"default": -1, "min": -1, "max": MAX_SAFE_INT}),
                "hop_length": ("INT", {"default": -1, "min": -1, "max": MAX_SAFE_INT}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("SPEC",)
    RETURN_NAMES = ("melspec",)
    FUNCTION = "mel_spectrogram"

    def mel_spectrogram(
        self,
        audio: AudioData,
        n_fft,
        n_mels,
        win_length=None,
        hop_length=None,
    ):
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=audio["sample_rate"],
            n_fft=n_fft,
            win_length=win_length if win_length != -1 else None,
            hop_length=hop_length if hop_length != -1 else None,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
        )

        melspec = mel_spectrogram(audio["waveform"])
        return (SpectrogramData(melspec, audio["sample_rate"]),)


class MFCC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "n_mfcc": ("INT", {"default": 40, "min": 0, "max": MAX_SAFE_INT}),
                "n_fft": ("INT", {"default": 1024, "min": 0, "max": MAX_SAFE_INT}),
                "n_mels": ("INT", {"default": 256, "min": 0, "max": MAX_SAFE_INT}),
            },
            "optional": {
                "win_length": ("INT", {"default": -1, "min": -1, "max": MAX_SAFE_INT}),
                "hop_length": ("INT", {"default": -1, "min": -1, "max": MAX_SAFE_INT}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("SPEC",)
    RETURN_NAMES = ("spec",)
    FUNCTION = "MFCC"

    def MFCC(
        self,
        audio: AudioData,
        n_mfcc,
        n_fft,
        n_mels,
        win_length=None,
        hop_length=None,
    ):
        mfcc_transform = T.MFCC(
            sample_rate=audio["sample_rate"],
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "win_length": win_length if win_length != -1 else None,
                "hop_length": hop_length if hop_length != -1 else None,
                "mel_scale": "htk",
                "n_mels": n_mels,
            },
        )

        mfcc = mfcc_transform(audio["waveform"])
        return (SpectrogramData(mfcc, audio["sample_rate"]),)


class LFCC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "n_filter": ("INT", {"default": 128, "min": 0, "max": MAX_SAFE_INT}),
                "n_lfcc": ("INT", {"default": 40, "min": 0, "max": MAX_SAFE_INT}),
                "n_fft": ("INT", {"default": 1024, "min": 0, "max": MAX_SAFE_INT}),
            },
            "optional": {
                "win_length": ("INT", {"default": -1, "min": -1, "max": MAX_SAFE_INT}),
                "hop_length": ("INT", {"default": 512, "min": -1, "max": MAX_SAFE_INT}),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("SPEC",)
    RETURN_NAMES = ("spec",)
    FUNCTION = "LFCC"

    def LFCC(
        self,
        audio: AudioData,
        n_filter,
        n_lfcc,
        n_fft,
        win_length=None,
        hop_length=None,
    ):
        mfcc_transform = T.LFCC(
            sample_rate=audio["sample_rate"],
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft,
                "win_length": win_length if win_length != -1 else None,
                "hop_length": hop_length if hop_length != -1 else None,
            },
        )

        lfcc = mfcc_transform(audio["waveform"])
        return (SpectrogramData(lfcc, audio["sample_rate"]),)


NODE_CLASS_MAPPINGS = {
    "SDT_Spectrogram": Spectrogram,
    "SDT_GriffinLim": GriffinLim,
    "SDT_MelSpectrogram": MelSpectrogram,
    "SDT_MFCC": MFCC,
    "SDT_LFCC": LFCC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_Spectrogram": "Spectrogram",
    "SDT_GriffinLim": "GriffinLim",
    "SDT_MelSpectrogram": "MelSpectrogram",
    "SDT_MFCC": "MFCC",
    "SDT_LFCC": "LFCC",
}
