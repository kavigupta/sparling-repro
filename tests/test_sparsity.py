import io
import os
import re
import sys
import tempfile
import unittest

import numpy as np
import torch

from latex_decompiler.dataset import DATA_TYPE_MAP
from latex_decompiler.experiments import e2e_dataset_1
from latex_decompiler.latex_cfg import latex_cfg
from latex_decompiler.train import train_latex_e2e
from latex_decompiler.utils import construct

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DIR = os.path.join(REPO_ROOT, "tests", "reference_data")


def _small_sparse_architecture():
    return dict(
        type="LaTeXPredictor",
        channels=8,
        motifs_spec=dict(type="ExtractorCNN"),
        sparsity_spec=dict(
            type="SparseLayerWithBatchNorm",
            underlying_sparsity_spec=dict(type="EnforceSparsityPerChannel2D"),
            affine=True,
            starting_sparsity=0.5,
        ),
        encoder_spec=dict(
            type="RowLSTMTransformerEncoder",
            row_lstm_spec=dict(type="BasicLSTM", bidirectional=True),
            transformer_encoder_spec=dict(type="TransformerEncoder", nhead=1, layers=1),
        ),
        decoder_spec=dict(
            type="TransformerCFGDecoder",
            cfg=latex_cfg,
            transformer_decoder_spec=dict(type="TransformerDecoder", nhead=1, layers=1),
        ),
    )


AGGRESSIVE_SUO = dict(
    type="LinearThresholdAdaptiveSUO",
    initial_threshold=0,
    minimal_threshold=-100,
    maximal_threshold=1,
    threshold_decrease_per_iter=0.01,
    minimal_update_frequency=0,
    information_multiplier=0.75,
)


def _train_with_sparsity(
    *, total_steps=1000, val_every=100, done_at_density=-float("inf")
):
    torch.manual_seed(0)
    return train_latex_e2e(
        path=tempfile.mkdtemp(),
        architecture=_small_sparse_architecture(),
        data_spec=e2e_dataset_1(),
        train_seed=1,
        val_seed=2,
        batch_size=2,
        total_steps=total_steps,
        print_every=10000,
        val_every=val_every,
        device="cpu",
        suo_spec=AGGRESSIVE_SUO,
        done_at_density=done_at_density,
    )


def _strip_dates(text):
    return re.sub(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\]", "[DATE]", text)


def _capture_train(**kwargs):
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        model = _train_with_sparsity(**kwargs)
    finally:
        sys.stdout = old_stdout
    return model, _strip_dates(buf.getvalue())


class SparsityTrainingTest(unittest.TestCase):
    def test_sparsity_increases(self):
        model, output = _capture_train()
        # Starting sparsity is 0.5. With 10 validation calls each multiplying
        # density by 0.75, final density = 0.5 * 0.75^10 ≈ 0.028, sparsity ≈ 0.97
        self.assertGreater(model.sparsity_value, 0.9)
        ref = np.load(os.path.join(REFERENCE_DIR, "sparsity_full.npz"))
        model.eval()
        with torch.no_grad():
            output = model.run_motifs_without_post_sparse(torch.tensor(ref["input"]))
        # Sparse outputs diverge more across environments because values near
        # the sparsity threshold can flip to/from zero.
        out = output.numpy()
        ref_out = ref["output"]
        mean_abs_diff = np.mean(np.abs(out - ref_out))
        self.assertLess(mean_abs_diff, 0.01)

    def test_done_at_density(self):
        model, output = _capture_train(done_at_density=0.1)
        density = 1 - model.sparsity_value
        # Training should stop when density drops below 0.1.
        # After 6 SUO updates: density = 0.5 * 0.75^6 ≈ 0.089 < 0.1
        self.assertLess(density, 0.1)
        # But not too many more updates beyond that
        self.assertGreater(density, 0.01)
        ref = np.load(os.path.join(REFERENCE_DIR, "sparsity_done_at_density.npz"))
        model.eval()
        with torch.no_grad():
            output = model.run_motifs_without_post_sparse(torch.tensor(ref["input"]))
        out = output.numpy()
        ref_out = ref["output"]
        mean_abs_diff = np.mean(np.abs(out - ref_out))
        self.assertLess(mean_abs_diff, 0.01)

    def test_sparsity_empirical(self):
        model, output = _capture_train()
        self.assertEqual(EXPECTED_TRAINING_OUTPUT, output)
        model.eval()

        dset = construct(DATA_TYPE_MAP, e2e_dataset_1(), seed=1)
        xs = torch.tensor([dset[i][0] for i in range(20)]).float()

        with torch.no_grad():
            sparse_output = model.run_motifs_without_post_sparse(xs)

        empirical_density = (sparse_output != 0).float().mean().item()
        reported_density = 1 - model.sparsity_value
        self.assertAlmostEqual(reported_density, 2.8156757355e-2, places=5)

        # Empirical density should be within a factor of 2 of the reported density.
        print(
            f"Empirical density: {empirical_density}, Reported density: {reported_density}"
        )
        self.assertGreater(reported_density, empirical_density * 0.5)
        self.assertLess(reported_density, empirical_density * 2)
        self.assertAlmostEqual(empirical_density, 0.0428, places=2)
        self.assertAlmostEqual(reported_density, 0.0282, places=3)


EXPECTED_TRAINING_OUTPUT = """\
starting at step 0
Accuracy: 0.00%
100 0 {'acc': 0.0}
Accuracy: 0.00%; Threshold: -100.00%
Originally using information (1 - sparsity) = 50.0000000000%
Now        using information (1 - sparsity) = 37.5000000000%
Actual   : y SUB( 4 SUB) 3 SUP( x SUP) + 2 SUP( x SUP) + PAREN( 3 PAREN) a SUP( 0 SUP)
Predicted: FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2
Accuracy: 0.00%
200 0.0 {'acc': 0.0}
Accuracy: 0.00%; Threshold: -100.00%
Originally using information (1 - sparsity) = 37.5000000000%
Now        using information (1 - sparsity) = 28.1250000000%
Actual   : y SUB( 4 SUB) 3 SUP( x SUP) + 2 SUP( x SUP) + PAREN( 3 PAREN) a SUP( 0 SUP)
Predicted: FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2
Accuracy: 0.00%
300 0.0 {'acc': 0.0}
Accuracy: 0.00%; Threshold: -100.00%
Originally using information (1 - sparsity) = 28.1250000000%
Now        using information (1 - sparsity) = 21.0937500000%
Actual   : y SUB( 4 SUB) 3 SUP( x SUP) + 2 SUP( x SUP) + PAREN( 3 PAREN) a SUP( 0 SUP)
Predicted: FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 9 FRACMID FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) FRAC) 9 FRACMID FRAC) 2 FRAC) 2 FRAC) 9 FRACMID FRAC) 2
Accuracy: 0.00%
400 0.0 {'acc': 0.0}
Accuracy: 0.00%; Threshold: -100.00%
Originally using information (1 - sparsity) = 21.0937500000%
Now        using information (1 - sparsity) = 15.8203125000%
Actual   : y SUB( 4 SUB) 3 SUP( x SUP) + 2 SUP( x SUP) + PAREN( 3 PAREN) a SUP( 0 SUP)
Predicted: 5 FRACMID 9 SUP( FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2
Accuracy: 0.00%
500 0.0 {'acc': 0.0}
Accuracy: 0.00%; Threshold: -100.00%
Originally using information (1 - sparsity) = 15.8203125000%
Now        using information (1 - sparsity) = 11.8652343750%
Actual   : y SUB( 4 SUB) 3 SUP( x SUP) + 2 SUP( x SUP) + PAREN( 3 PAREN) a SUP( 0 SUP)
Predicted: 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 2 FRAC) 9 FRACMID 9 8
Accuracy: 0.00%
600 0.0 {'acc': 0.0}
Accuracy: 0.00%; Threshold: -100.00%
Originally using information (1 - sparsity) = 11.8652343750%
Now        using information (1 - sparsity) = 8.8989257812%
Actual   : y SUB( 4 SUB) 3 SUP( x SUP) + 2 SUP( x SUP) + PAREN( 3 PAREN) a SUP( 0 SUP)
Predicted: 2 + FRAC) 2 + + + + + + + + + + + + + + + + + + + + + + + FRAC) 2 +
Accuracy: 0.00%
700 0.0 {'acc': 0.0}
Accuracy: 0.00%; Threshold: -100.00%
Originally using information (1 - sparsity) = 8.8989257812%
Now        using information (1 - sparsity) = 6.6741943359%
Actual   : y SUB( 4 SUB) 3 SUP( x SUP) + 2 SUP( x SUP) + PAREN( 3 PAREN) a SUP( 0 SUP)
Predicted: 2 + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
Accuracy: 0.00%
800 0.0 {'acc': 0.0}
Accuracy: 0.00%; Threshold: -100.00%
Originally using information (1 - sparsity) = 6.6741943359%
Now        using information (1 - sparsity) = 5.0056457520%
Actual   : y SUB( 4 SUB) 3 SUP( x SUP) + 2 SUP( x SUP) + PAREN( 3 PAREN) a SUP( 0 SUP)
Predicted: 2 + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
Accuracy: 0.00%
900 0.0 {'acc': 0.0}
Accuracy: 0.00%; Threshold: -100.00%
Originally using information (1 - sparsity) = 5.0056457520%
Now        using information (1 - sparsity) = 3.7542343140%
Actual   : y SUB( 4 SUB) 3 SUP( x SUP) + 2 SUP( x SUP) + PAREN( 3 PAREN) a SUP( 0 SUP)
Predicted: PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN(
Accuracy: 0.00%
1000 0.0 {'acc': 0.0}
Accuracy: 0.00%; Threshold: -100.00%
Originally using information (1 - sparsity) = 3.7542343140%
Now        using information (1 - sparsity) = 2.8156757355%
Actual   : y SUB( 4 SUB) 3 SUP( x SUP) + 2 SUP( x SUP) + PAREN( 3 PAREN) a SUP( 0 SUP)
Predicted: 2 PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( PAREN( 2 PAREN( PAREN( PAREN( PAREN( PAREN( 2 PAREN( PAREN( PAREN( PAREN( PAREN( 2 PAREN( PAREN(
"""
