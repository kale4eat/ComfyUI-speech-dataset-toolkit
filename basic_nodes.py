import glob
import os
import random

import torchaudio

import folder_paths

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
        return (AudioData(waveform, sample_rate), file_name)


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
            [AudioData(item[0], item[1]) for item in items],
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
        if "." not in file_name:
            file_name = file_name + _AUDIO_FILE_FORMAT_EXT_MAP[file_format]
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        file_name = (
            os.path.basename(os.path.normpath(filename_prefix)) + "_" + file_name
        )
        full_output_folder = os.path.join(self.output_dir, subfolder)
        os.makedirs(full_output_folder, exist_ok=True)
        file = os.path.join(self.output_dir, file_name)
        torchaudio.save(file, audio.waveform, audio.sample_rate, format=file_format)
        return {}


class SaveAudioWithSequentialNumbering:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
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
        file_format: str,
        filename_prefix: str,
    ):
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_util.get_audio_output_directory()
        )
        file = os.path.join(
            full_output_folder,
            f"{filename}_{counter:05}_{_AUDIO_FILE_FORMAT_EXT_MAP[file_format]}",
        )
        torchaudio.save(file, audio.waveform, audio.sample_rate, format=file_format)
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
    RETURN_NAMES = ("sampl_rate",)
    FUNCTION = "prop"
    OUTPUT_NODE = True

    def prop(self, audio: AudioData):
        return (audio.sample_rate,)


class PlayAudio:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_temp_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audios": ("AUDIO",),
            },
        }

    CATEGORY = BASE_NODE_CATEGORY
    INPUT_IS_LIST = (True,)
    RETURN_TYPES = ()
    FUNCTION = "play_audio"
    OUTPUT_NODE = True

    def play_audio(self, audios: list[AudioData]):
        max_counts = 50
        results = []
        for audio in audios[:max_counts]:
            filename_prefix = "_temp_" + "".join(
                random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5)
            )
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(filename_prefix, self.output_dir)
            )

            filename = f"{filename}.wav"
            torchaudio.save(
                os.path.join(full_output_folder, filename),
                audio.waveform,
                audio.sample_rate,
                format="wav",
            )
            results.append(
                [
                    {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": "temp",
                    }
                ]
            )
        return {
            "ui": {"audios": results},
        }


NODE_CLASS_MAPPINGS = {
    "SDT_LoadAudio": LoadAudio,
    "SDT_LoadAudios": LoadAudios,
    "SDT_SaveAudio": SaveAudio,
    "SDT_SaveAudioWithSequentialNumbering": SaveAudioWithSequentialNumbering,
    "SDT_AudioProperty": AudioProperty,
    "SDT_PlayAudio": PlayAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_LoadAudio": "Load Audio",
    "SDT_LoadAudios": "Load Audios",
    "SDT_SaveAudio": "Save Audio",
    "SDT_SaveAudioWithSequentialNumbering": "Save Audio With Sequential Numbering",
    "SDT_AudioProperty": "Audio Property",
    "SDT_PlayAudio": "Play Audio",
}
