import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from permacache import no_cache_global

from pixel_art.display.perturbations import compute_perturbation_accuracy

COUNT = 50
CONFUSION_SAMPLES = 100


class TestPerturbationAccuracy(unittest.TestCase):
    def test_digit_circle(self):
        with no_cache_global():
            acc = compute_perturbation_accuracy(
                COUNT,
                key="digit_circle",
                confusion_num_samples=CONFUSION_SAMPLES,
            )
        self.assertGreater(acc, 0.9)

    def test_latex_ocr(self):
        with no_cache_global():
            acc = compute_perturbation_accuracy(
                COUNT,
                key="latex_ocr",
                confusion_num_samples=CONFUSION_SAMPLES,
            )
        self.assertGreater(acc, 0.7)

    def test_audio_mnist_sequence(self):
        with no_cache_global():
            acc = compute_perturbation_accuracy(
                COUNT,
                key="audio_mnist_sequence",
                confusion_num_samples=CONFUSION_SAMPLES,
            )
        self.assertGreater(acc, 0.8)
