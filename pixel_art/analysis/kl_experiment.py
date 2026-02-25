import os

import torch
from permacache import permacache

from latex_decompiler.remapping_pickle import load_with_remapping_pickle
from latex_decompiler.utils import construct, load_model
from pixel_art.analysis.evaluate_motifs import evaluate_motifs
from pixel_art.domain.domain import domain_types
from pixel_art.domain.stamp import digit_stamps

SELECTED_CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "selected_checkpoints",
)

kl_names = {
    0.01: "pae-9baf1_1",
    0.1: "pae-9bai1_1",
    1: "pae-9bal1_1",
    10: "pae-9bao1_1",
    100: "pae-9bar1_1",
    1000: "pae-9bau1_1",
    10_000: "pae-9bax1_1",
    100_000: "pae-9baza1_1",
}

data_spec = dict(
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
)


def _load_kl_model(model_name, step):
    path = os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, f"step_{step}")
    return load_with_remapping_pickle(path, weights_only=False).eval()


def get_step_for_model(model_name):
    """Read the step from the selected_checkpoints directory for a KL model."""
    ckpt_dir = os.path.join(SELECTED_CHECKPOINTS_DIR, model_name)
    step_files = [f for f in os.listdir(ckpt_dir) if f.startswith("step_")]
    assert len(step_files) == 1, f"Expected 1 step file in {ckpt_dir}, got {step_files}"
    return int(step_files[0].split("_")[1])


def get_val_accuracy(model_name, step):
    """Load validation accuracy from the model/ directory."""
    path = f"model/{model_name}"
    _, v = load_model(path, step, key="val_result")
    return 100 * v["val_fn_result"]["accuracy"]


@permacache("pixel_art/analysis/kl_experiment/compute_motif_density")
def compute_motif_density(model_name, step, samples=1000):
    """Load a KL model and compute motif activation statistics.

    Returns (mean_activation, density) where density = fraction of activations > 0.5.
    """
    actual = construct(domain_types(), data_spec["domain_spec"])
    stamps = digit_stamps()
    model = _load_kl_model(model_name, step)
    _, _, mots, _ = evaluate_motifs(model, actual, stamps, samples=samples)
    return float(mots.mean()), float((mots > 0.5).mean())


def compute_all_kl_results():
    """Compute density and val accuracy for all KL models.

    Returns dict mapping lambda -> (step, val_accuracy, mean_activation, density).
    """
    results = {}
    for lam, name in kl_names.items():
        ckpt_dir = os.path.join(SELECTED_CHECKPOINTS_DIR, name)
        if not os.path.isdir(ckpt_dir):
            continue
        step = get_step_for_model(name)
        val_acc = get_val_accuracy(name, step)
        mean_act, density = compute_motif_density(name, step)
        results[lam] = dict(
            step=step,
            val_acc=val_acc,
            mean_activation=100 * mean_act,
            density=100 * density,
        )
    return results
