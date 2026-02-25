import torch

from pixel_art.audo_mnist.spectrogram.torchaudio_transforms import (
    AmplitudeToDB,
    MelSpectrogram,
)


def spectro_gram(aud, *, n_mels, n_fft=1024, hop_len=None):
    """
    From https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
    """
    sig, sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = AmplitudeToDB(top_db=top_db)(spec)
    return spec


def construct_spectrogram(audio, *, n_mels, time_chunk, sample_rate):
    n_fft = int(2 * time_chunk * sample_rate / n_mels)
    tens = spectro_gram(
        (torch.tensor(audio).float(), sample_rate), n_mels=n_mels, n_fft=n_fft
    )
    return tens.numpy()
