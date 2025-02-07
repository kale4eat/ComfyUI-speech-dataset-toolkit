import torch


def is_monaural(waveform: torch.Tensor):
    return waveform.shape[1] == 1

def is_stereo(waveform):
    return waveform.shape[1] == 2

def stereo_to_monaural(waveform: torch.Tensor):
    return waveform.mean(dim=1, keepdim=True)

def monaural_to_pseudo_stereo(waveform: torch.Tensor):
    return torch.cat([waveform, waveform], dim=1)
