import itertools
import os

import numpy as np
import torch
import tqdm.auto as tqdm
from matplotlib import pyplot as plt
from permacache import CacheMissError, permacache

from latex_decompiler.dataset import DATA_TYPE_MAP
from latex_decompiler.latex_cfg import LATEX_CFG_SPECS
from latex_decompiler.remapping_pickle import load_with_remapping_pickle
from latex_decompiler.utils import construct
from pixel_art.analysis.evaluate_latex_motifs import (
    confusion_error,
    precisely_evaluate_latex_motifs_from_checkpoint_no_tags,
    unified_confusion_matrix,
)
from pixel_art.analysis.evaluate_motifs import (
    confusion_from_results,
    display_confusion,
    realign_confusion,
)
from pixel_art.analysis.main_experiment import SparsityBar

from .gather_evaluation import compute_all_errors, compute_statistic

SELECTED_CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "selected_checkpoints",
)

models = {
    "QBNLaTeX-32c": ("ltx-2dc3", 9, 32),
    # "EBNLaTeX-32c": ("ltx-3dc3", 5, 32),
}

w = 360
h = 120
pad = 2

spec = LATEX_CFG_SPECS["latex_cfg"]
latex_dset_spec = dict(
    type="LaTeXDataset",
    latex_cfg=spec["cfg"],
    font="computer_modern",
    data_config=dict(
        minimal_length=1,
        maximal_length=spec["maximal_length"],
        dpi=200,
        w=w,
        h=h,
    ),
)


def noise_latex_dset_spec(noise_amount):
    spec = latex_dset_spec.copy()
    spec["noise_amount"] = noise_amount
    spec["type"] = "NoisyBinaryLaTeXDataset"
    return spec


def test_data():
    return construct(DATA_TYPE_MAP, latex_dset_spec, seed=-2)


@permacache("pixel_art/analysis/experiment/mean_num_characters")
def mean_num_characters():
    dset = test_data()
    return np.mean(
        [
            len([x for tok in dset[i][1] for x in tok.rendered_symbols])
            for i in range(200)
        ]
    )


def sparsity_bar(num_motifs):
    mean_chars = mean_num_characters()
    return SparsityBar.right_above_line(mean_chars / (num_motifs * h * w), num_motifs)


def density_above_line(bar, times_above_line):
    return bar.sparsity_bar / 0.75**times_above_line


def step_above_line(bar, path, seed, times_above_line):
    density = density_above_line(bar, times_above_line)
    model_name = path + "_" + str(seed)
    checkpoint_key = f"{density:.6e}"
    if not os.path.exists(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key)
    ):
        return density, model_name, None
    return density, model_name, checkpoint_key


def load_model_above_line(model, seed, times_above_line):
    path, _, num_motifs = models[model]
    _, model_name, checkpoint_key = step_above_line(
        sparsity_bar(num_motifs), path, seed, times_above_line
    )
    if checkpoint_key is None:
        return None
    return load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()


def all_results(
    *,
    max_above_line,
    num_samples,
    models=models,
    latex_dset_spec=latex_dset_spec,
    monitoring_mode=False,
):
    fn = precisely_evaluate_latex_motifs_from_checkpoint_no_tags
    kwargs = dict(dset_spec=latex_dset_spec, num_samples=num_samples, pad=pad)

    return all_results_generic(
        fn,
        kwargs,
        models=models,
        max_above_line=max_above_line,
        monitoring_mode=monitoring_mode,
        compute_sparsity_bar=sparsity_bar,
    )


def all_results_generic(
    fn, kwargs, *, models, max_above_line, monitoring_mode, compute_sparsity_bar
):
    results = {}
    for times_above_line, model in list(
        itertools.product(range(1 + max_above_line), models)
    ):
        results_each = []
        for seed in range(1, 1 + models[model][1] + 1):
            path, _, num_motifs = models[model]
            _, model_name, checkpoint_key = step_above_line(
                compute_sparsity_bar(num_motifs),
                path,
                seed,
                times_above_line=times_above_line,
            )
            if checkpoint_key is None:
                continue
            if monitoring_mode:
                try:
                    with fn.error_on_miss():
                        result = fn(model_name, checkpoint_key, **kwargs)
                except CacheMissError:
                    continue
            else:
                result = fn(model_name, checkpoint_key, **kwargs)
            results_each.append(result)
        results[model, times_above_line] = results_each
    return results


def display_grouped_confusion(overall, size=10):
    display_confusion(*realign_confusion(unified_confusion_matrix(overall)), size=size)


def compute_ces(models, results):
    return compute_statistic(confusion_error, models, results)


def plot_ces(ces, limit=True):
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        warnings.filterwarnings(
            action="ignore", message="invalid value encountered in double_scalars"
        )
        max_above_line = max(len(x) - 1 for x in ces.values())
        plt.axhline(2.5, color="black")
        for m in ces:
            plt.plot(
                np.arange(1 + max_above_line),
                [100 * np.mean(np.array(x)) for x in ces[m]],
                label=m,
            )
        plt.legend()
        plt.ylabel("Confusion error [%]")
        plt.xlabel("Number of density steps above line")
        plt.xlim(plt.xlim()[1], plt.xlim()[0])
        if limit:
            plt.ylim(0, 10)
        plt.grid()
        plt.show()


def all_latex_errors(max_num_above_line):
    model_to_evaluate = "QBNLaTeX-32c"
    return compute_all_errors(
        model_to_evaluate,
        models[model_to_evaluate],
        motif_results_fn=lambda **kwargs: all_results(
            **kwargs, latex_dset_spec=noise_latex_dset_spec(0.25)
        ),
        dset_spec=noise_latex_dset_spec(0.25),
        sparsity_bar_fn=sparsity_bar,
        max_num_above_line=max_num_above_line,
        retrain_path="ltx-4dc3",
        retrain_seeds=9,
        retrain_step=60_000,
    )


def latex_confusion_matrix(num_samples=10_000):
    results_qbn = all_results(
        max_above_line=0,
        num_samples=num_samples,
        models=models,
        latex_dset_spec=noise_latex_dset_spec(0.25),
        monitoring_mode=False,
    )
    res = results_qbn["QBNLaTeX-32c", 0][0]
    return unified_confusion_matrix(res, rename="join")
