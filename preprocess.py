import torch
import torchaudio
import torchaudio.transforms as T
from random import uniform


def load_audio(file_path, target_sample_rate=44100):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = T.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    return waveform


def induce_clipping(waveform, file_path):
    # Randomize gain amount
    gain_amt = uniform(1.1, 1.4)

    # Add gain to file until it clips
    iterations = 0
    gain_up = T.Vol(gain=gain_amt, gain_type="amplitude")
    while torch.max(waveform) < 1.0 and iterations < 100:
        waveform = gain_up(waveform)
        iterations += 1

    if iterations >= 100:
        print(file_path)
        raise Exception("Problem with waveform clipping, exceeded max iterations.")

    # Hard clip the waveform
    clipped_waveform = torch.clip(waveform, -1.0, 1.0)
    return clipped_waveform


def waveform_to_spectrogram(waveform, n_fft=1024, hop_length=512):
    spectrogram_transform = T.MelSpectrogram(n_fft=n_fft, hop_length=hop_length)
    return spectrogram_transform(waveform)


def process_audio(file_path):
    waveform = load_audio(file_path)
    clipped_waveform = induce_clipping(waveform, file_path)

    spec_original = waveform_to_spectrogram(waveform)
    spec_clipped = waveform_to_spectrogram(clipped_waveform)

    return spec_clipped, spec_original

