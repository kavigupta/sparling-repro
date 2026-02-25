import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from permacache import stable_hash

from latex_decompiler.dataset import DATA_TYPE_MAP
from latex_decompiler.experiments import e2e_dataset_1
from latex_decompiler.latex_cfg import LATEX_CFG_SPECS, latex_cfg
from latex_decompiler.train import train_latex_e2e
from latex_decompiler.utils import construct, load_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DIR = os.path.join(REPO_ROOT, "tests", "reference_data")


def hash_state_dict(model):
    return stable_hash(list(model.state_dict().values()))


def run_experiment(experiment_path, seed="1"):
    tmpdir = tempfile.mkdtemp()
    try:
        full_experiment_path = os.path.join(REPO_ROOT, experiment_path)
        with open(full_experiment_path) as f:
            source = f.read()

        patch = (
            "import torch\n"
            "from permacache import no_cache_global\n"
            "torch.manual_seed(0)\n"
            "exp.total_steps = 2\n"
            "exp.batch_size = 2\n"
            "exp.val_every = 1\n"
            "exp.device = 'cpu'\n"
            f"exp._model_checkpoint_path_override = {tmpdir!r}\n"
            "with no_cache_global():\n"
            "    exp.run()\n"
        )
        source = source.replace("exp.run()", patch)

        tmp_script = os.path.join(tmpdir, "run.py")
        with open(tmp_script, "w") as f:
            f.write(source)

        result = subprocess.run(
            [sys.executable, tmp_script, seed],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Subprocess failed with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        _, checkpoint = load_model(tmpdir)
        assert checkpoint is not None, f"No model checkpoint found in {tmpdir}"
        return checkpoint["model"]
    finally:
        shutil.rmtree(tmpdir)


class DataRegressionTest(unittest.TestCase):
    def test_dset_regression(self):
        dset = construct(DATA_TYPE_MAP, e2e_dataset_1(), seed=1)
        images = np.stack([dset[i][0] for i in range(100)])
        ref = np.load(os.path.join(REFERENCE_DIR, "dset_regression.npz"))["images"]
        mean_abs_diff = np.mean(np.abs(images - ref))
        self.assertAlmostEqual(mean_abs_diff, 0, places=3)

    def test_dset_regression_pixelart(self):
        dset = construct(
            DATA_TYPE_MAP,
            dict(
                type="PixelArtDomainDataset",
                domain_spec=dict(
                    type="StampCircleDomain",
                    size=100,
                    min_radius=20,
                    random_shift=3,
                    max_syms=6,
                    pre_noise=0.5,
                    post_noise=0.05,
                ),
                stamps_spec=dict(type="digit_stamps"),
            ),
            seed=1,
        )
        self.assertEqual(
            "8b6eee4a61a268198a90836d579dda238b235869ab2291b0588f795431aff532",
            stable_hash([dset[i] for i in range(0, 100)]),
        )

    def test_dset_regression_latex(self):
        spec = LATEX_CFG_SPECS["latex_cfg"]
        dset = construct(
            DATA_TYPE_MAP,
            dict(
                type="NoisyBinaryLaTeXDataset",
                latex_cfg=spec["cfg"],
                font="computer_modern",
                data_config=dict(
                    minimal_length=1,
                    maximal_length=spec["maximal_length"],
                    dpi=200,
                    w=360,
                    h=120,
                ),
                noise_amount=0.25,
            ),
            seed=1,
        )
        images = np.stack([dset[i][0] for i in range(100)])
        ref = np.load(os.path.join(REFERENCE_DIR, "dset_regression_latex.npz"))[
            "images"
        ]
        frac_different = np.mean(images != ref)
        self.assertAlmostEqual(frac_different, 0, places=3)

    def test_dset_regression_audiomnist(self):
        dset = construct(
            DATA_TYPE_MAP,
            dict(
                type="AudioMNISTDomainDataset",
                domain_spec=dict(
                    type="AudioClipDomain",
                    digits_per_speaker_limit=None,
                    clip_length_seconds=15,
                    length_range=(5, 10),
                    speaker_set=list(range(1, 1 + 51)),
                    noise_amplitude=-10,
                    operation_spec=dict(type="ListOffDigitsOperation"),
                ),
                n_mels=64,
            ),
            seed=1,
        )
        self.assertEqual(
            "981c60354d0df9906e2c05aa6f0a53ae2d84403345a0d3f3ec8e688c2bdca519",
            stable_hash([dset[i] for i in range(0, 100)]),
        )

    def test_dset_regression_splice(self):
        dset = construct(
            DATA_TYPE_MAP,
            dict(type="SplicingDataset", is_training=True),
            seed=1,
        )
        self.assertEqual(
            "c1502a9fcd914dca4f24a4390b1e7d24354543ccf75b0b6b771c03cc17925938",
            stable_hash([dset[i] for i in range(0, 100)]),
        )

    def test_training_regression(self):
        torch.manual_seed(0)
        model = train_latex_e2e(
            path=tempfile.mkdtemp(),
            architecture=dict(
                type="LaTeXPredictor",
                channels=8,
                motifs_spec=dict(type="ExtractorCNN"),
                sparsity_spec=dict(type="NoSparsity", starting_sparsity=0),
                encoder_spec=dict(
                    type="RowLSTMTransformerEncoder",
                    row_lstm_spec=dict(type="BasicLSTM", bidirectional=True),
                    transformer_encoder_spec=dict(
                        type="TransformerEncoder", nhead=1, layers=1
                    ),
                ),
                decoder_spec=dict(
                    type="TransformerCFGDecoder",
                    cfg=latex_cfg,
                    transformer_decoder_spec=dict(
                        type="TransformerDecoder", nhead=1, layers=1
                    ),
                ),
            ),
            data_spec=e2e_dataset_1(),
            train_seed=1,
            val_seed=2,
            batch_size=2,
            total_steps=2,
            print_every=1000,
            val_every=1000,
            device="cpu",
        )
        h = stable_hash(list(model.state_dict().values()))
        self.assertEqual(
            "f0c9fd89599447842183060c92573c7934effcd926c591ea2658e04a3ed5414b", h
        )

    def test_training_regression_digitcircle(self):
        state = run_experiment("pixel_art/experiments/pae-9baf1.py")
        self.assertEqual(
            "6da3affde77b130c28289c3a32fb5c5fbb8d3c1948f010f90145b808deda795c",
            hash_state_dict(state),
        )

    def test_training_regression_audiomnist(self):
        state = run_experiment("pixel_art/experiments/aum-2a1.py")
        self.assertEqual(
            "27f37b46109f6b21527d3ee6d2b0a5f2bf6473c2955c3cd75e10f3a4b8396d04",
            hash_state_dict(state),
        )

    def test_training_regression_splice(self):
        state = run_experiment("pixel_art/experiments/spl-1c2.py")
        self.assertEqual(
            "fb9b6949cef8f2258c27921eb7e18a667409e03d4083fb89bd6ae7a1a8f2a238",
            hash_state_dict(state),
        )

    def test_training_regression_main_pixelart(self):
        state = run_experiment("pixel_art/experiments/pae-7bb1.py")
        self.assertEqual(
            "e1b87e017dd1ea5ac189c21f03da5070f47436f5cf281bd21319755b9dc22e96",
            hash_state_dict(state),
        )

    def test_training_regression_main_latex(self):
        state = run_experiment("pixel_art/experiments/ltx-2dc3.py")
        self.assertEqual(
            "c1992de32529d89fab502a5d0b2e969e35aa8e80bd64129e0435c425708c62d1",
            hash_state_dict(state),
        )

    def test_training_regression_main_audiomnist(self):
        model = run_experiment("pixel_art/experiments/aum-2ka1.py")
        ref = np.load(
            os.path.join(REFERENCE_DIR, "training_regression_main_audiomnist.npz")
        )
        x = torch.tensor(ref["input"])
        model.eval()
        with torch.no_grad():
            output = model.run_motifs_without_post_sparse(x)
        if isinstance(output, dict):
            output = output["motifs_for_loss"]
        np.testing.assert_allclose(
            output.cpu().numpy(),
            ref["output"],
            atol=0.05,
            rtol=0.01,
            err_msg="Model behavior mismatch on reference input",
        )
