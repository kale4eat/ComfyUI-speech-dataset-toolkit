# BS-RoFormer
# https://github.com/lucidrains/BS-RoFormer

import torch
import torch.nn as nn
import torchaudio
import yaml
from bs_roformer import BSRoformer, MelBandRoformer
from ml_collections import ConfigDict

from ..node_def import BASE_NODE_CATEGORY, AudioData

NODE_CATEGORY = BASE_NODE_CATEGORY + "/ai/BS-RoFormer"


class BSRoFormerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_check_point": ("STRING", {"default": ""}),
                "config_path": ("STRING", {"default": ""}),
                "device": (["auto", "cpu", "cuda"],),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("BS_ROFORMER",)
    RETURN_NAMES = ("bs_rformer",)
    FUNCTION = "load"

    def load(
        self,
        start_check_point: str,
        config_path: str,
        device: str,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(config_path) as f:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

        assert isinstance(config.model, ConfigDict)
        param = dict(config.model)
        # if "linear_transformer_depth" in param:
        #     del param["linear_transformer_depth"]
        model = BSRoformer(**param)

        state_dict = torch.load(
            start_check_point, map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict)
        model = model.to(device)
        return ((model, config),)


class MelBandRoformerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_check_point": ("STRING", {"default": ""}),
                "config_path": ("STRING", {"default": ""}),
                "device": (["auto", "cpu", "cuda"],),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("BS_ROFORMER",)
    RETURN_NAMES = ("bs_rformer",)
    FUNCTION = "load"

    def load(
        self,
        start_check_point: str,
        config_path: str,
        device: str,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(config_path) as f:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

        assert isinstance(config.model, ConfigDict)
        param = dict(config.model)
        model = MelBandRoformer(**param)

        state_dict = torch.load(
            start_check_point, map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict)
        model = model.to(device)
        return ((model, config),)


def _get_windowing_array(window_size, fade_size):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window


def _demix(config, model, mix: torch.Tensor, device):
    C = config.audio.chunk_size
    N = config.inference.num_overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = config.inference.batch_size if hasattr(config.inference, "batch_size") else 4

    length_init = mix.shape[-1]

    # Do pad from the beginning and end to account floating window results better
    if length_init > 2 * border and (border > 0):
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    # windowing_array crossfades at segment boundaries to mitigate clicking artifacts
    windowing_array = _get_windowing_array(C, fade_size)

    use_amp = config.training.use_amp if hasattr(config.training, "use_amp") else True
    with torch.autocast(device_type=device.type, enabled=use_amp):
        with torch.inference_mode():
            if config.training.target_instrument is not None:
                req_shape = (1,) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []

            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i : i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(
                            input=part, pad=(0, C - length), mode="reflect"
                        )
                    else:
                        part = nn.functional.pad(
                            input=part,
                            pad=(0, C - length, 0, 0),
                            mode="constant",
                            value=0,
                        )
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    window = windowing_array
                    if i - step == 0:  # First audio chunk, no fadein
                        window[:fade_size] = 1
                    elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                        window[-fade_size:] = 1

                    for j in range(len(batch_locations)):
                        start, length_ = batch_locations[j]
                        result[..., start : start + length_] += (
                            x[j][..., :length_].cpu() * window[..., :length_]
                        )
                        counter[..., start : start + length_] += window[..., :length_]

                    batch_data = []
                    batch_locations = []

            estimated_sources = result / counter
            estimated_sources = torch.nan_to_num(estimated_sources, nan=0.0)

            if length_init > 2 * border and (border > 0):
                # Remove pad
                estimated_sources = estimated_sources[..., border:-border]

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return {
            k: v for k, v in zip([config.training.target_instrument], estimated_sources)
        }


class BSRoFormerApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BS_ROFORMER",),
                "audio": ("AUDIO",),
            },
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("vocals", "other")
    FUNCTION = "run"

    def run(
        self,
        model,
        audio: AudioData,
    ) -> tuple[AudioData, AudioData]:
        model_, config = model

        model_input_wave = audio.waveform
        if audio.sample_rate != config.audio.sample_rate:
            transform = torchaudio.transforms.Resample(
                orig_freq=audio.sample_rate, new_freq=config.audio.sample_rate
            )

            model_input_wave = transform(model_input_wave)

        if not audio.is_stereo():
            model_input_wave = torch.cat([model_input_wave, model_input_wave], dim=0)

        waveforms = _demix(
            config,
            model_,
            model_input_wave,
            next(model_.parameters()).device,
        )

        instruments = config.training.instruments.copy()
        if config.training.target_instrument is not None:
            instruments = [config.training.target_instrument]
        instr = "vocals" if "vocals" in instruments else instruments[0]
        instruments.append("instrumental")
        # Output "instrumental", which is an inverse of 'vocals' or the first stem in list if 'vocals' absent
        waveforms["instrumental"] = model_input_wave - waveforms[instr]

        # NOTE: Resample is not done here for extensibility
        return (
            AudioData(waveforms["vocals"], config.audio.sample_rate),
            AudioData(waveforms["instrumental"], config.audio.sample_rate),
        )


NODE_CLASS_MAPPINGS = {
    "SDT_BSRoFormerLoader": BSRoFormerLoader,
    "SDT_MelBandRoformerLoader": MelBandRoformerLoader,
    "SDT_BSRoFormerApply": BSRoFormerApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDT_BSRoFormerLoader": "Load BS-RoFormer",
    "SDT_MelBandRoformerLoader": "Load MelBandRoformer",
    "SDT_BSRoFormerApply": "Apply BS-RoFormer",
}
