# ComfyUI-speech-dataset-toolkit

## Overview

Basic audio tools using torchaudio for ComfyUI. It is assumed to assist in the speech dataset creation for ASR, TTS, etc.

## Features

- Basic
    - Load & Save audio
- Edit
    - Cut and Trim
    - Split and Join
    - Silence
    - Resample
- Visualization
    - WaveForm
    - Specgram
    - Spectrogram
    - MelFilterBank
    - Pitch
- AI
    - [Demucs (facebook)](https://github.com/facebookresearch/demucs)
    - [faster-whisper (OpenAI's Whisper model using CTranslate2)](https://github.com/SYSTRAN/faster-whisper)
    - [silero-vad (Silero Team)](https://github.com/snakers4/silero-vad)
    - [nue-asr (rinna Co., Ltd.)](https://github.com/rinnakk/nue-asr)
    - [ReazonSpeech nemo-asr (Reazon Human Interaction Lab)](https://github.com/reazon-research/ReazonSpeech/tree/master/pkg/nemo-asr)
    - [SpeechMOS (UTokyo-SaruLab's MOS prediction system)](https://github.com/tarepan/SpeechMOS)
    - [kotoba-whisper (Kotoba Technologies)](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0)
    - [BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)

## Requirement

Install torchaudio according to your environment.

```bash
cd custom_nodes
git clone https://github.com/kale4eat/ComfyUI-speech-dataset-toolkit.git
cd ComfyUI-speech-dataset-toolkit
pip3 install torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
```

If you use silero-vad, install onnxruntime according to your environment.
```bash
pip install onnxruntime-gpu
```

## Usage

At first startup, `audio_input` and `audio_output` folder is created. 

```
ComfyUI
├── audio_input
├── audio_output
├── custom_nodes
│   └── ComfyUI-speech-dataset-toolkit
...
```

Fisrt of all, use a `Load Audio` node to load audio.

![Load Audio node](images/load_audio_node.png)

Please put the audio files you wish to process in a `audio_input` folder in advance.
If you've added files while the app is running, please reload the page (press F5).

`audio`, the data type of ComfyUI flow, consists of waveform and sample rate.
Many nodes of this extension handle this data.

For example, Demucs separate drums, bass, vocals and other stems. Each of them is audio data.

![Apply demucs node](images/apply_demucs_node.png)

Finally, use a `Save Audio` node to save audio. The audio is saved to `audio_output` folder.

![Save Audio node](images/save_audio_node.png)

## Note

There are some unsettled policies, destructive changes may be made.

This repository does not contain the nodes such as numerical operations and string processing.

## Inspiration

- [ComfyUI-audio](https://github.com/eigenpunk/ComfyUI-audio)
- [ComfyUI-AudioScheduler](https://github.com/a1lazydog/ComfyUI-AudioScheduler)
