import glob
from functools import lru_cache

import librosa
import numpy as np
from permacache import permacache

SAMPLE_RATE = 8000


digits = range(10)
speakers = range(1, 1 + 60)

speaker_order = list(speakers)
np.random.RandomState(0).shuffle(speaker_order)


# audio_paths[digit][speaker] -> list of audio paths
@lru_cache(None)
def audio_paths():
    return {
        digit: {
            speaker: sorted(
                glob.glob(f"AudioMNIST/data/{speaker_file:02d}/{digit}_*.wav")
            )
            for speaker, speaker_file in zip(speakers, speaker_order)
        }
        for digit in digits
    }


def audio_sample_paths(digit, speaker):
    return audio_paths()[digit][speaker]


def random_audio_sample(digit, speaker_set, rng, limit=None):
    speaker = rng.choice(speaker_set)
    return rng.choice(audio_sample_paths(digit, speaker)[:limit])


@permacache("pixel_art/audio_mnist/data/samples/from_path_4")
def from_path(path):
    return librosa.load(path, sr=SAMPLE_RATE)[0]
