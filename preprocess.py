from typing import Any
import os
import torch
from torchaudio import save
import torchaudio.transforms as T
import torchcodec
from random import uniform
import glob
from pathlib import Path


def chunk_dataset(data_dir, chunk_dir, chunk_size) -> None:
    os.makedirs(chunk_dir, exist_ok=True)
    files = glob.glob(f"{data_dir}/*.mp3")

    for file in files:
        print("Processing", file)
        waveform = process_audio(file, return_waveform_only=True)
        length = waveform.shape[-1]
        stem = Path(file).stem

        for i, start in enumerate(range(0, length - chunk_size, chunk_size)):
            chunk = waveform[:, start:start + chunk_size]
            out = f"{chunk_dir}/{stem}_{i}.mp3"
            save(out, chunk, 44100)


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

