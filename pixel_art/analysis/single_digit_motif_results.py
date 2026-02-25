import os

import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm
from matplotlib import pyplot as plt
from permacache import drop_if_equal, permacache

from latex_decompiler.dataset import DATA_TYPE_MAP
from latex_decompiler.remapping_pickle import load_with_remapping_pickle
from latex_decompiler.utils import construct
from pixel_art.analysis.audio_mnist_experiment import digits_data as audio_digits_data
from pixel_art.analysis.latex_digit_dataset import digits_data as latex_digits_data
from pixel_art.analysis.pixel_art_experiment import digits_data as pixel_art_digits_data

SELECTED_CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "selected_checkpoints",
)


def _density_key(density):
    return f"{density:.6e}"


def _step_key(step):
    return f"step_{step}"


def _checkpoint_exists(model_name, key):
    return os.path.exists(os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, key))


def _load_checkpoint(model_name, key):
    path = os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, key)
    return load_with_remapping_pickle(path, weights_only=False).eval()


def _errors_for_checkpoint_key(model_spec, checkpoint_key, *, spec, mask, **kwargs):
    p, max_seed = model_spec
    accs = {}
    for seed in range(1, 1 + max_seed):
        model_name = f"{p}_{seed}"
        if not _checkpoint_exists(model_name, checkpoint_key):
            continue
        matr = compute_matrix_for_checkpoint(model_name, checkpoint_key, spec, **kwargs)
        accs[seed] = compute_accuracy(matr, mask=mask)
    return 100 - pd.Series(accs) * 100


def single_digit_motifs_errors_at_density(model_spec, density, *, spec, mask, **kwargs):
    return _errors_for_checkpoint_key(
        model_spec, _density_key(density), spec=spec, mask=mask, **kwargs
    )


def compute_motifs_non_clipped(mod, xs):
    chunk = 128
    mots = []
    for i in range(0, xs.shape[0], chunk):
        with torch.no_grad():
            mot = mod.run_motifs_without_post_sparse(
                torch.tensor(xs[i : i + chunk, :, :]).cuda(), disable_relu=True
            )
            mot = mot.cpu().numpy()
            mots.append(mot)
    mots = np.concatenate(mots)
    return mots


def compute_classifications(mod, xs):
    mots = compute_motifs_non_clipped(mod, xs)
    while len(mots.shape) > 2:
        mots = mots.max(axis=-1)
    return mots.argmax(axis=-1)


def compute_classification_matrix(mod, xs, ys, *, num_pred_classes, num_true_classes):
    """
    Returns matr[pred_class, true_class]
    """
    classes = compute_classifications(mod, xs)
    matr = np.zeros((num_pred_classes, num_true_classes))
    np.add.at(matr, (classes, ys), 1)
    return matr


def compute_accuracy(matr, mask):
    # matr[pred_class, true_class]
    # each pred class corresponds to some true class
    true_class_for_pred_class = matr.argmax(1)
    if mask is not None:
        pred_class_mask = mask[true_class_for_pred_class]
        matr = matr[pred_class_mask][:, mask]
        return compute_accuracy(matr, mask=None)
    corresponding = matr[range(matr.shape[0]), true_class_for_pred_class]
    return corresponding.sum() / matr.sum()


@permacache(
    "pixel_art/analysis/single_digit_motif_results_3",
    key_function=dict(
        num_pred_classes=drop_if_equal(10), num_true_classes=drop_if_equal(10)
    ),
)
def compute_matrix_for_checkpoint(
    model_name, checkpoint_key, load_data_spec, *, num_pred_classes, num_true_classes
):
    print("Computing matrix for", model_name, checkpoint_key, "...")
    xs, ys = construct(
        dict(
            audio_data=audio_digits_data,
            latex_data=latex_digits_data,
            pixel_art_data=pixel_art_digits_data,
        ),
        load_data_spec,
    )
    mod = _load_checkpoint(model_name, checkpoint_key)
    return compute_classification_matrix(
        mod,
        xs,
        ys,
        num_pred_classes=num_pred_classes,
        num_true_classes=num_true_classes,
    )


def single_digit_motifs_errors_at_density_for_steps(
    model_spec, dset_spec, steps, *, dimensions, mask
):
    direct_errs_at_step = {}
    for step in steps:
        direct_errs_at_step[step] = _errors_for_checkpoint_key(
            model_spec, _step_key(step), spec=dset_spec, mask=mask, **dimensions
        )
    return pd.DataFrame(direct_errs_at_step)


def plot_sparling_vs_direct_model(errs, direct_errs_at_step):
    for k in direct_errs_at_step:
        [x] = plt.plot(
            list(direct_errs_at_step[k]),
            [direct_errs_at_step[k][step].mean() for step in direct_errs_at_step[k]],
            label=f"{k} [direct]",
            linestyle="--",
        )
        plt.axhline(errs[k].mean(), color=x._color, label=f"{k} [sparling]")
        # plt.axhline(errs[k].median(), color=x._color, label=f"{k} [sparling]")
    plt.legend()
    plt.grid()


def all_direct_eval(
    sparling_model_spec,
    sparling_model_density,
    direct_model_spec,
    direct_model_steps,
    dsets,
    *,
    num_model_classes=10,
    num_real_classes=10,
    mask=None,
):
    errs = {
        k: single_digit_motifs_errors_at_density(
            sparling_model_spec,
            sparling_model_density,
            spec=dsets[k],
            num_pred_classes=num_model_classes,
            num_true_classes=num_real_classes,
            mask=mask,
        )
        for k in dsets
    }
    direct_errs_at_step = {}
    for k in dsets:
        direct_errs_at_step[k] = single_digit_motifs_errors_at_density_for_steps(
            direct_model_spec,
            dsets[k],
            direct_model_steps,
            dimensions=dict(
                num_pred_classes=num_real_classes, num_true_classes=num_real_classes
            ),
            mask=mask,
        )

    return errs, direct_errs_at_step


def compute_digits_data(dset_spec, seed, amount):
    dset = construct(DATA_TYPE_MAP, dset_spec, seed=seed)
    audios, digits = [], []
    for i in tqdm.trange(amount):
        audio, digit = dset[i]
        audios.append(audio)
        digits.append(digit)
    audios = np.array(audios)
    return audios, digits


def summary_table(all_results):
    out = {}
    for domain, (err_sparling, err_direct) in all_results.items():
        assert set(err_sparling) == set(err_direct)
        for setting in err_sparling:
            name = domain
            if len(err_sparling) > 1:
                name += "/" + setting
            last_step = max(list(err_direct[setting]))
            out[name] = {
                r"\textsc{Sparling} [mean]": err_sparling[setting].mean(),
                r"\textsc{Direct} [mean]": err_direct[setting][last_step].mean(),
            }
            out[name]["Ratio [of means]"] = (
                out[name][r"\textsc{Direct} [mean]"]
                / out[name][r"\textsc{Sparling} [mean]"]
            )
    return pd.DataFrame(out).T
