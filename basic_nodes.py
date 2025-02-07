import glob
import os

import torchaudio

from . import folder_util
from .node_def import BASE_NODE_CATEGORY, AudioData

_AUDIO_FILE_FORMAT = [
    "wav",
    "mp3",
    "flac",
    "vorbis",
    "sph",
    "amb",
    "amr-nb",
    "gsm",
]

_AUDIO_FILE_EXT = [
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".sph",
    ".amb",
    "amr",
    ".gsm",
]

_AUDIO_FILE_FORMAT_EXT_MAP = {
    "wav": ".wav",
    "mp3": ".mp3",
    "flac": ".flac",
    "vorbis": ".ogg",
    "sph": ".sph",
    "amb": ".amb",
    "amr-nb": ".amr",
    "gsm": ".gsm",
}


def _is_audio_file(f):
    return any(ext for ext in _AUDIO_FILE_EXT if f.lower().endswith(ext))


class LoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_util.get_audio_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        audio_files = [f for f in files if _is_audio_file(f)]
        return {
            "required": {"file_name": (sorted(audio_files), {"audio_upload": True})},
        }

    CATEGORY = BASE_NODE_CATEGORY
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_name")
    FUNCTION = "load_audio"

    def load_audio(self, file_name):
        file = os.path.join(folder_util.get_audio_input_directory(), file_name)
        waveform, sample_rate = torchaudio.load(file)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, file_name)


class LoadAudios:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_util.get_audio_input_directory()
        dirs = [
            os.path.relpath(d, input_dir)
            for d in glob.glob(input_dir + "/**", recursive=True)
            if os.path.isdir(d)
        ]
        return {
            "required": {"dir": (sorted(dirs), {"audio_upload": True})},
        }

    CATEGORY = BASE_NODE_CATEGORY
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audios", "file_names")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_audios"

    def load_audios(self, dir):
        abs_dir = os.path.join(folder_util.get_audio_input_directory(), dir)
        abs_dir = os.path.abspath(abs_dir)
        files = [
            f for f in os.listdir(abs_dir) if os.path.isfile(os.path.join(abs_dir, f))
        ]
        audio_files = [os.path.join(abs_dir, f) for f in files if _is_audio_file(f)]
        items = [torchaudio.load(f) for f in audio_files]
        return (
            [{"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate} for waveform, sample_rate in items],
            [os.path.basename(f) for f in audio_files],
        )

class SaveAudio:
    def __init__(self):
        self.output_dir = folder_util.get_audio_output_directory()
        self.output_type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "file_name": ("STRING", {"default": "audio"}),
                "file_format": (_AUDIO_FILE_FORMAT,),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
        }

    CATEGORY = BASE_NODE_CATEGORY
    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True

    def save_audio(
        self,
        audio: AudioData,
        file_name: str,
        file_format: str,
        filename_prefix: str,
    ):
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(self.output_dir, subfolder)
        os.makedirs(full_output_folder, exist_ok=True)

        if "." not in file_name:
            file_name = file_name + _AUDIO_FILE_FORMAT_EXT_MAP[file_format]

        for b in range(audio["waveform"].shape[0]):
            filename_with_batch_num = file_name.replace("%batch_num%", str(b))

            filename_with_batch_num = (
                os.path.basename(os.path.normpath(filename_prefix)) + "_" + filename_with_batch_num
            )

            file = os.path.join(self.output_dir, filename_with_batch_num)
            torchaudio.save(file, audio["waveform"][b], audio["sample_rate"], format=file_format)
        return {}


class AudioProperty:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    CATEGORY = BASE_NODE_CATEGORY
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("sample_rate",)
    FUNCTION = "prop"
    OUTPUT_NODE = True

    def prop(self, audio: AudioData):
        return (audio["sample_rate"],)

NODE_CLASS_MAPPINGS = {
    "SDT_LoadAudio": LoadAudio,
    "SDT_LoadAudios": LoadAudios,
    "SDT_SaveAudio": SaveAudio,
    "SDT_AudioProperty": AudioProperty,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_LoadAudio": "Load Audio",
    "SDT_LoadAudios": "Load Audios",
    "SDT_SaveAudio": "Save Audio",
    "SDT_AudioProperty": "Audio Property",
}
