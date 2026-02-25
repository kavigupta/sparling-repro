import attr
import numpy as np
import torch
from scipy import stats

from latex_decompiler.cfg import Token
from latex_decompiler.utils import compute_seed, construct
from pixel_art.audo_mnist.spectrogram.spectrogram import construct_spectrogram

from .samples import SAMPLE_RATE, from_path, random_audio_sample


class SummationOperation:
    def __call__(self, digits):
        return list(str(sum(digits)))


class SummationOperationLastDigit:
    def __call__(self, digits):
        return list(str(sum(digits) % 10))


class MaxOperation:
    def __call__(self, digits):
        return list(str(max(digits)))


class ListOffDigitsOperation:
    def __call__(self, digits):
        return [str(x) for x in digits]


def operation_types():
    return dict(
        SummationOperation=SummationOperation,
        SummationOperationLastDigit=SummationOperationLastDigit,
        MaxOperation=MaxOperation,
        ListOffDigitsOperation=ListOffDigitsOperation,
    )


@attr.s
class AudioClipDomain:
    digits_per_speaker_limit = attr.ib()
    clip_length_seconds = attr.ib()
    length_range = attr.ib()
    speaker_set = attr.ib()
    noise_amplitude = attr.ib()

    operation_spec = attr.ib()

    def sample_with_metadata(self, rng):
        length = rng.randint(self.length_range[0], self.length_range[1] + 1)

        digits = rng.choice(10, size=length)
        paths = [
            random_audio_sample(
                digit, self.speaker_set, rng, limit=self.digits_per_speaker_limit
            )
            for digit in digits
        ]
        digit_samples = [from_path(path) for path in paths]

        samples = int(SAMPLE_RATE * self.clip_length_seconds)
        audio = np.zeros(samples, np.float32)

        digit_clip_lengths = [x.shape[0] for x in digit_samples]
        digit_starts = self.compute_digit_starts(
            rng, digit_samples, digit_clip_lengths, audio.shape[0]
        )

        for digit_start, digit_sample in zip(digit_starts, digit_samples):
            audio[digit_start : digit_start + digit_sample.shape[0]] = digit_sample
        noise = stats.truncnorm(
            -1, 1, scale=min(2**16, 2**self.noise_amplitude)
        ).rvs(samples, random_state=rng)

        result = construct(operation_types(), self.operation_spec)(digits)

        return (
            audio + noise,
            result,
            dict(
                digits=digits,
                paths=paths,
                digit_starts=digit_starts,
                digit_clip_lengths=digit_clip_lengths,
            ),
        )

    def compute_digit_starts(
        self, rng, digit_samples, digit_clip_lengths, audio_length
    ):
        while True:
            digit_starts = rng.choice(audio_length, size=len(digit_samples))
            digit_starts.sort()
            spaces = [*digit_starts[1:], audio_length] - digit_starts
            if (spaces >= digit_clip_lengths).all():
                break
        return digit_starts

    def sample(self, rng):
        x, y, _ = self.sample_with_metadata(rng)
        return x, y


class AudioMNISTSingleDigitDomain(AudioClipDomain):
    def compute_digit_starts(
        self, rng, digit_samples, digit_clip_lengths, audio_length
    ):
        assert len(digit_samples) == 1
        [clip_length] = digit_clip_lengths
        return [(audio_length - clip_length) // 2]


def audio_domain_types():
    return dict(
        AudioClipDomain=AudioClipDomain,
        AudioMNISTSingleDigitDomain=AudioMNISTSingleDigitDomain,
    )


@attr.s
class AudioClipDomainSpectrogram:
    domain_spec = attr.ib()
    n_mels = attr.ib()
    time_chunk = attr.ib()

    def sample_with_metadata(self, rng, stamps):
        # stamps just taken for compatibility with pixel art
        del stamps

        domain = construct(audio_domain_types(), self.domain_spec)
        audio, symbols, metadata = domain.sample_with_metadata(rng)
        spec = construct_spectrogram(
            audio,
            n_mels=self.n_mels,
            time_chunk=self.time_chunk,
            sample_rate=SAMPLE_RATE,
        )
        spec = spec.T

        stamps = []

        for digit, digit_start, digit_clip_length in zip(
            metadata["digits"], metadata["digit_starts"], metadata["digit_clip_lengths"]
        ):
            digit_start = digit_start * spec.shape[0] // audio.shape[0]
            digit_clip_length = digit_clip_length * spec.shape[0] // audio.shape[0]

            point = digit_start + digit_clip_length // 2, spec.shape[1] // 2
            slices = (
                slice(digit_start, digit_start + digit_clip_length),
                slice(0, spec.shape[1]),
            )

            stamps.append(dict(point=point, symbol=str(digit), slices=slices))

        return spec.astype(np.float32), symbols, dict(**metadata, placed_stamps=stamps)

    def sample(self, rng):
        x, y, _ = self.sample_with_metadata(rng, stamps=None)
        return x, y


class AudioMNISTDomainDataset:
    def __init__(self, domain_spec, seed, *, looping=None, n_mels=64, time_chunk=1.5):
        self.domain_spec = domain_spec
        self.seed = seed
        self.looping = looping
        self.n_mels = n_mels
        self.time_chunk = time_chunk

    def __getitem__(self, idx):
        spec, symbols = AudioClipDomainSpectrogram(
            self.domain_spec, self.n_mels, self.time_chunk
        ).sample(np.random.RandomState(compute_seed(self.seed, idx, self.looping)))
        return spec, [Token.single_symbol(s) for s in symbols]

    @property
    def data_config(self):
        return dict(maximal_length=self.domain_spec["length_range"][1] + 3)


class AudioMNISTSingleDigitDomainDataset(AudioMNISTDomainDataset):
    def __getitem__(self, idx):
        x, [y] = super().__getitem__(idx)
        return x, int(y.name)
