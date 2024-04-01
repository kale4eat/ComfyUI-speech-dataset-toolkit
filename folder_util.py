import os

import folder_paths

_audio_input_directory = os.path.join(
    os.path.dirname(os.path.realpath(folder_paths.__file__)), "audio_input"
)

_audio_output_directory = os.path.join(
    os.path.dirname(os.path.realpath(folder_paths.__file__)), "audio_output"
)


def get_audio_input_directory():
    global _audio_input_directory
    return _audio_input_directory


def get_audio_output_directory():
    global _audio_output_directory
    return _audio_output_directory


def initiarize():
    os.makedirs(get_audio_input_directory(), exist_ok=True)
    os.makedirs(get_audio_output_directory(), exist_ok=True)
