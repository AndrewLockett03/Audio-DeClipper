from typing import Any

import torch
import torchaudio.transforms as T
import torchcodec
from random import uniform


def load_audio(file_path, target_sample_rate=44100):
    waveform = torchcodec.decoders.AudioDecoder(source=file_path, sample_rate=target_sample_rate)

    return waveform, target_sample_rate


def induce_clipping(waveform, sample_rate):

    # How many samples to count (the loudest half second or entire file if shorter)
    samples = waveform.get_all_samples()

    # Grab the loudest n samples and clip to the mean of them
    abs_samples = torch.abs(samples.data[0])
    top_samples = torch.topk(abs_samples, k=min(int(sample_rate/4), abs_samples.numel())).values
    clip_value = torch.mean(top_samples)

    # Add up to 5% randomness in the clip value
    # bounds = clip_value.item() / 20
    # clip_value = clip_value + uniform(-bounds, bounds)
    clip_value = clip_value * (1 + uniform(-0.05, 0.05))

    # Hard clip the waveform
    clipped_waveform = torch.clip(waveform.get_all_samples().data, -clip_value, clip_value)

    return clipped_waveform


def waveform_to_spectrogram(waveform, sample_rate, n_fft=1024, hop_length=512):
    spectrogram_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length)
    return spectrogram_transform(waveform)


def process_audio(file_path, return_waveform_only=False):
    print("preprocessing audio chunk")
    decoder, sample_rate = load_audio(file_path)

    waveform_original = decoder.get_all_samples().data  # (channels, samples)
    if return_waveform_only:
        return waveform_original

    waveform_clipped = induce_clipping(decoder, sample_rate)

    # spec_original = waveform_to_spectrogram(waveform_original, sample_rate)
    # spec_clipped = waveform_to_spectrogram(waveform_clipped, sample_rate)
    # return clipped_waveform
    spec_original = Any
    spec_clipped = Any

    return spec_clipped, spec_original, waveform_clipped, waveform_original

