import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from permacache import no_cache_global

from latex_decompiler.evaluate import accuracy_at_checkpoint
from latex_decompiler.utils import construct
from pixel_art.analysis.audio_mnist_experiment import (
    ac_multi_speaker_noisy_domain_test,
    precisely_evaluate_audio_mnist_motifs_untagged_from_checkpoint,
)
from pixel_art.analysis.evaluate_latex_motifs import (
    confusion_error,
    false_negative_error,
    false_positive_error,
    precisely_evaluate_latex_motifs_from_checkpoint_no_tags,
)
from pixel_art.analysis.evaluate_motifs import errors_from_checkpoint
from pixel_art.analysis.latex_experiment import noise_latex_dset_spec, pad
from pixel_art.analysis.main_experiment import data_spec as pixel_art_data_spec
from pixel_art.domain.domain import domain_types
from pixel_art.domain.stamp import digit_stamps


class TestAccuracyAtCheckpoint(unittest.TestCase):
    def test_density_based_pixel_art(self):
        with no_cache_global():
            acc = accuracy_at_checkpoint(
                "pae-7bb1_1",
                "5.022621e-05",
                pixel_art_data_spec,
                "edit-dist",
                batch_size=100,
            )
        self.assertAlmostEqual(acc, 0.9942861904761905, places=4)

    def test_density_based_pixel_art_high_density(self):
        with no_cache_global():
            acc = accuracy_at_checkpoint(
                "pae-7bb1_1",
                "5.016956e-04",
                pixel_art_data_spec,
                "edit-dist",
                batch_size=100,
            )
        self.assertAlmostEqual(acc, 0.9980709523809524, places=4)

    def test_step_based_retrain(self):
        with no_cache_global():
            acc = accuracy_at_checkpoint(
                "pae-7bbb1_1",
                "step_600000",
                pixel_art_data_spec,
                "edit-dist",
                batch_size=100,
            )
        self.assertAlmostEqual(acc, 0.9981064285714286, places=4)

    def test_non_annealing_1_5x(self):
        with no_cache_global():
            acc = accuracy_at_checkpoint(
                "pae-7bbw2_1",
                "step_9020000",
                pixel_art_data_spec,
                "edit-dist",
                batch_size=100,
            )
        self.assertAlmostEqual(acc, 1 - 71.0175 / 100, places=4)

    def test_non_annealing_1_1x(self):
        with no_cache_global():
            acc = accuracy_at_checkpoint(
                "pae-7bbx2_1",
                "step_9020000",
                pixel_art_data_spec,
                "edit-dist",
                batch_size=100,
            )
        self.assertAlmostEqual(acc, 1 - 68.115167 / 100, places=4)

    def test_no_batchnorm_st(self):
        with no_cache_global():
            acc = accuracy_at_checkpoint(
                "pae-2b2_1",
                "5.022621e-05",
                pixel_art_data_spec,
                "edit-dist",
                batch_size=100,
            )
        self.assertAlmostEqual(acc, 1 - 62.2957 / 100, places=4)

    def test_no_batchnorm_mt(self):
        with no_cache_global():
            acc = accuracy_at_checkpoint(
                "pae-2ba2_1",
                "5.022621e-05",
                pixel_art_data_spec,
                "edit-dist",
                batch_size=100,
            )
        self.assertAlmostEqual(acc, 1 - 73.126 / 100, places=4)


class TestDigitCircleMotifs(unittest.TestCase):
    def test_errors_from_checkpoint(self):
        actual = construct(domain_types(), pixel_art_data_spec["domain_spec"])
        stamps = digit_stamps()
        with no_cache_global():
            fne, fpe, ce = errors_from_checkpoint(
                "pae-7bb1_1", "5.022621e-05", actual, stamps, samples=500
            )
        self.assertAlmostEqual(fne, 0.002234137622877569, places=4)
        self.assertAlmostEqual(fpe, 0.0013416815742397137, places=4)
        self.assertAlmostEqual(ce, 0.0004478280340349805, places=4)

    def test_errors_from_checkpoint_high_density(self):
        actual = construct(domain_types(), pixel_art_data_spec["domain_spec"])
        stamps = digit_stamps()
        with no_cache_global():
            fne, fpe, ce = errors_from_checkpoint(
                "pae-7bb1_1", "5.016956e-04", actual, stamps, samples=500
            )
        self.assertAlmostEqual(fne, 0, places=4)
        self.assertAlmostEqual(fpe, 0.10012062726176116, places=4)
        self.assertAlmostEqual(ce, 0.4343163538873994, places=4)


class TestLatexMotifs(unittest.TestCase):
    def test_from_checkpoint(self):
        with no_cache_global():
            result = precisely_evaluate_latex_motifs_from_checkpoint_no_tags(
                "ltx-2dc3_1",
                "8.939187e-06",
                noise_latex_dset_spec(0.25),
                num_samples=100,
                pad=pad,
            )
        self.assertAlmostEqual(confusion_error(result), 0.022641509433962263, places=4)
        self.assertAlmostEqual(
            false_negative_error(result), 0.1657922350472193, places=4
        )
        self.assertAlmostEqual(
            false_positive_error(result), 0.031668696711327646, places=4
        )

    def test_from_checkpoint_high_density(self):
        with no_cache_global():
            result = precisely_evaluate_latex_motifs_from_checkpoint_no_tags(
                "ltx-2dc3_1",
                "8.929105e-05",
                noise_latex_dset_spec(0.25),
                num_samples=100,
                pad=pad,
            )
        self.assertAlmostEqual(confusion_error(result), 0.25867861142217247, places=4)
        self.assertAlmostEqual(
            false_negative_error(result), 0.06295907660020986, places=4
        )
        self.assertAlmostEqual(
            false_positive_error(result), 0.23609923011120615, places=4
        )


class TestAudioMNISTMotifs(unittest.TestCase):
    def test_from_checkpoint(self):
        with no_cache_global():
            result = precisely_evaluate_audio_mnist_motifs_untagged_from_checkpoint(
                "aum-2ka1_1",
                "1.189204e-03",
                ac_multi_speaker_noisy_domain_test,
                amount=100,
            )
        self.assertAlmostEqual(confusion_error(result), 0.01912568306010929, places=4)
        self.assertAlmostEqual(
            false_negative_error(result), 0.02529960053262317, places=4
        )
        self.assertAlmostEqual(false_positive_error(result), 0.0, places=4)

    def test_from_checkpoint_high_density(self):
        with no_cache_global():
            result = precisely_evaluate_audio_mnist_motifs_untagged_from_checkpoint(
                "aum-2ka1_1",
                "8.908974e-03",
                ac_multi_speaker_noisy_domain_test,
                amount=100,
            )
        self.assertAlmostEqual(confusion_error(result), 0.4447403462050599, places=4)
        self.assertAlmostEqual(false_negative_error(result), 0.0, places=4)
        self.assertAlmostEqual(
            false_positive_error(result), 0.6545538178472861, places=4
        )
