import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from permacache import no_cache_global

from pixel_art.analysis.entropy_per_nonzero_activation import (
    compute_binned_accuracy_from_checkpoint,
)

high_sparsity = "5.022621e-05"
low_sparsity = "5.016956e-04"


class TestBinnedAccuracy(unittest.TestCase):
    def checkBinnedComputation(self, checkpoint_key, bins, expected_err):
        with no_cache_global():
            acc = compute_binned_accuracy_from_checkpoint(
                "pae-7bb1_1",
                checkpoint_key,
                (
                    dict(type="to_percentiles", num_bins=bins)
                    if bins is not None
                    else dict(type="identity")
                ),
                calibration_samples=100,
                evaluation_samples=100,
            )
        self.assertAlmostEqual(acc, 1 - expected_err / 100, places=4)

    def test_identity_binner(self):
        self.checkBinnedComputation(high_sparsity, None, 0.53)

    def test_identity_binner_high_density(self):
        self.checkBinnedComputation(low_sparsity, None, 0.29)

    def test_percentile_binner_4_bins(self):
        self.checkBinnedComputation(high_sparsity, 4, 1.84)

    def test_percentile_binner_4_bins_high_density(self):
        self.checkBinnedComputation(low_sparsity, 4, 2.03)

    def test_percentile_binner_2_bins(self):
        self.checkBinnedComputation(high_sparsity, 2, 0.79)

    def test_percentile_binner_2_bins_high_density(self):
        self.checkBinnedComputation(low_sparsity, 2, 2.98)
