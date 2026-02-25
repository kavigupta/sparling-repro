import os

import pandas as pd
import torch
from permacache import drop_if_equal, permacache

from latex_decompiler.evaluate import SELECTED_CHECKPOINTS_DIR, accuracy_at_checkpoint
from latex_decompiler.remapping_pickle import load_with_remapping_pickle
from latex_decompiler.utils import construct
from pixel_art.analysis.evaluate_motifs import compute_confusion_matrix, errors_from_all
from pixel_art.analysis.evaluate_retrain import collect_retrain_results_for_step
from pixel_art.analysis.main_experiment import (
    accuracies_at_density,
    data_spec,
    load_sparsity_bars,
)
from pixel_art.domain.domain import domain_types
from pixel_art.domain.stamp import digit_stamps

models = {
    "ST": ("pae-6bb1", 9),
    "MT": ("pae-7bb1", 9),
}

single_digit_spec = dict(
    type="PixelArtDomainSingleDigitDataset",
    domain_spec=dict(
        type="StampCircleDomainSingleDigit", size=32, pre_noise=0.5, post_noise=0.05
    ),
    stamps_spec=dict(type="digit_stamps"),
)


def sparsity(steps_above):
    return load_sparsity_bars().right_above_line.sparsity_bar / 0.75**steps_above


def end_to_end_errors(steps_above, *, accuracy_metric):
    df_full = 100 - accuracies_at_density(
        models, data_spec, sparsity(steps_above), accuracy_metric=accuracy_metric
    )
    return df_full.T


def all_pixel_art_errors_above_line(steps_above):
    stamps = digit_stamps()

    actual = construct(domain_types(), data_spec["domain_spec"])
    fne, fpe, ce = errors_from_all(
        models,
        sparsity(steps_above),
        actual,
        stamps,
    )
    # e2e_exact = end_to_end_errors(steps_above, accuracy_metric="exact")
    e2e_edit = end_to_end_errors(steps_above, accuracy_metric="edit-dist")

    return {
        mod: pd.DataFrame(
            dict(
                fne=fne[mod],
                fpe=fpe[mod],
                ce=ce[mod],
                # e2e_exact=e2e_exact[mod],
                e2e_edit=e2e_edit[mod],
            )
        )
        for mod in models
    }


def all_pixel_art_errors(max_num_above_line):
    result = {
        steps_above: all_pixel_art_errors_above_line(steps_above)
        for steps_above in range(1 + max_num_above_line)
    }
    result["not-sparse"] = {
        "ST": evaluate_non_sparse_model("pae-7bba1"),
        "MT": evaluate_non_sparse_model("pae-7bba1"),
    }
    result["sparse-retrained"] = {
        name: pd.DataFrame(
            dict(
                e2e_edit=100
                - collect_retrain_results_for_step(
                    path,
                    9,
                    data_spec=data_spec,
                    step=600_000,
                    quiet=True,
                    batch_size=100,
                )
            )
        )
        for name, path in [
            ("ST", "pae-6bbb1"),
            ("MT", "pae-7bbb1"),
        ]
    }

    return result


@permacache(
    "pixel_art/analysis/pixel_art_experiment/pixel_art_confusion_matrix_2",
    key_function=dict(num_samples=drop_if_equal(10_000)),
)
def pixel_art_confusion_matrix(num_samples=10_000):
    actual = construct(domain_types(), data_spec["domain_spec"])
    model_name = models["MT"][0] + "_1"
    density = load_sparsity_bars().right_above_line.sparsity_bar
    checkpoint_key = f"{density:.6e}"
    updated = load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()
    return compute_confusion_matrix(
        updated, actual, digit_stamps(), samples=num_samples, handle_multi=False
    )


def evaluate_non_sparse_model(model_path_prefix, step=600_000):
    checkpoint_key = f"step_{step}"
    res = {}
    for i in range(1, 1 + 9):
        model_name = f"{model_path_prefix}_{i}"
        if not os.path.exists(
            os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key)
        ):
            continue
        res[i] = accuracy_at_checkpoint(
            model_name,
            checkpoint_key,
            data_spec,
            "edit-dist",
            batch_size=100,
        )
    return 100 - 100 * pd.DataFrame(dict(e2e_edit=pd.Series(res)))


@permacache("pixel_art/analysis/pixel_art_experiment/digits_data_2")
def digits_data(dset_spec, seed, amount=10**4):
    from .single_digit_motif_results import compute_digits_data

    return compute_digits_data(dset_spec, seed, amount)
