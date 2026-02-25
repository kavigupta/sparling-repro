import unittest

import numpy as np
from permacache import no_cache_global

from pixel_art.analysis.audio_mnist_experiment import audio_mnist_data_spec_train
from pixel_art.analysis.latex_digit_dataset import latex_digits_spec
from pixel_art.analysis.pixel_art_experiment import (
    single_digit_spec as pixel_art_single_digit_spec,
)
from pixel_art.analysis.single_digit_motif_results import (
    compute_accuracy,
    compute_matrix_for_checkpoint,
)

AMOUNT = 500

pixel_art_test_spec = dict(
    type="pixel_art_data", dset_spec=pixel_art_single_digit_spec, seed=-2, amount=AMOUNT
)
latex_test_spec = dict(
    type="latex_data",
    dset_spec={**latex_digits_spec, "num_samples": 500},
    seed=-2,
    amount=AMOUNT,
)
audio_test_spec = dict(
    type="audio_data", dset_spec=audio_mnist_data_spec_train, seed=-2, amount=AMOUNT
)


class DirectEvalTest(unittest.TestCase):
    def test_digit_circle_matrix(self):
        with no_cache_global():
            matr = compute_matrix_for_checkpoint(
                "pae-11bb1_1",
                "step_1000000",
                pixel_art_test_spec,
                num_pred_classes=10,
                num_true_classes=10,
            )
        self.assertEqual(matr.sum(), AMOUNT)
        expected = np.array(
            [
                [53, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 59, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 39, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 50, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 48, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 45, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 59, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 45, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 54, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 48],
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(matr, expected)
        self.assertAlmostEqual(float(compute_accuracy(matr, mask=None)), 1.0, places=4)

    def test_latex_matrix(self):
        with no_cache_global():
            matr = compute_matrix_for_checkpoint(
                "ltx-5dc3_1",
                "step_3000000",
                latex_test_spec,
                num_pred_classes=19,
                num_true_classes=19,
            )
        self.assertEqual(matr.sum(), AMOUNT)
        expected = np.array(
            [
                [35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 26, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34],
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(matr, expected)
        self.assertAlmostEqual(
            float(compute_accuracy(matr, mask=None)), 0.996, places=3
        )

    def test_audio_mnist_matrix(self):
        with no_cache_global():
            matr = compute_matrix_for_checkpoint(
                "aum-4ka1_1",
                "step_3000000",
                audio_test_spec,
                num_pred_classes=10,
                num_true_classes=10,
            )
        self.assertEqual(matr.sum(), AMOUNT)
        expected = np.array(
            [
                [45, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 55, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 55, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 47, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 38, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 48, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 44, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 64, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 59],
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(matr, expected)
        self.assertAlmostEqual(
            float(compute_accuracy(matr, mask=None)), 0.994, places=3
        )
