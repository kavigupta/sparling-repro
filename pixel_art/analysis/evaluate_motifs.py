import os
import string
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import tqdm.auto as tqdm
from more_itertools import chunked
from permacache import permacache, stable_hash

from latex_decompiler.remapping_pickle import load_with_remapping_pickle

SELECTED_CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "selected_checkpoints",
)


def evaluate_motifs(m, domain, stamps, samples=10):
    actual_samples = [
        domain.sample_with_metadata(np.random.RandomState(i), stamps)
        for i in tqdm.trange(samples)
    ]
    xs = np.array([x for x, _, _ in actual_samples])
    ys = [y for _, y, _ in actual_samples]
    placed_stampses = [meta["placed_stamps"] for _, _, meta in actual_samples]
    mots = []
    with torch.no_grad():
        for xbs in tqdm.tqdm(list(chunked(xs, 10))):
            motif_out = m.run_motifs_without_post_sparse(
                torch.tensor(np.array(list(xbs))).float().cuda()
            )
            if isinstance(motif_out, dict):
                motif_out = motif_out["result"]
            mots.extend(motif_out.cpu().numpy())
    return xs, ys, np.array(mots), placed_stampses


def categorize_relationships(
    mot,
    placed_stamps,
    *,
    handle_multi,
    motif_names=string.ascii_uppercase,
    include_stamp=False,
):
    mot = mot.copy()
    assert mot.shape[0] <= len(motif_names)
    results = []

    for stamp in placed_stamps:
        slices = (slice(None),) + stamp["slices"]
        max_axes = tuple(range(1, len(slices)))
        by_category = mot[slices].max(max_axes)
        mot[slices] = 0
        if (by_category != 0).sum() == 0:
            pred = "none"
        elif handle_multi and (by_category != 0).sum() > 1:
            pred = motif_names[by_category.argmax()] + "*"
        else:
            pred = motif_names[by_category.argmax()]
        results.append([stamp, pred, stamp["symbol"]])

    for mot_id in np.where(mot)[0]:
        results.append([None, motif_names[mot_id], "none"])

    if not include_stamp:
        results = [(m, r) for _, m, r in results]
    return results


@permacache(
    "pixel_art/analysis/evaluate_motifs/compute_confusion_matrix_7",
    key_function=dict(m=stable_hash, domain=stable_hash, stamps=stable_hash),
)
def compute_confusion_matrix(m, domain, stamps, samples, *, handle_multi):
    _, _, mots, placed_stampses = evaluate_motifs(m, domain, stamps, samples=samples)
    results = []
    for mot, placed_stamps in zip(mots, placed_stampses):
        results.extend(
            categorize_relationships(mot, placed_stamps, handle_multi=handle_multi)
        )
    return confusion_from_results(results)


def confusion_from_results(results):
    confusion = defaultdict(lambda: defaultdict(int))
    for pred, true in results:
        confusion[pred][true] += 1
    confusion = pd.DataFrame(dict(confusion)).fillna(0)
    confusion = confusion.loc[
        sorted(confusion.index, key=lambda x: x if x != "none" else "zzz")
    ]
    confusion = confusion[sorted(confusion)]
    return confusion


def realign_confusion(confusion):
    per_row = confusion.sum(axis=1)
    confusion = confusion / np.array(confusion.sum(axis=1))[:, None]
    reordered = reorder_confusion(confusion)
    return reordered, per_row


def reorder_confusion(confusion):
    # sort columns such that the ones that contribute to each row
    # are in the same order as the rows
    confusion_wo_none = confusion.loc[[x for x in confusion.index if x != "none"]]
    column_ordering = {
        x: (confusion_wo_none[x].argmax(), confusion_wo_none[x].max())
        for x in confusion
    }
    column_ordering["none"] = float("inf"), float("inf")
    columns = sorted(confusion, key=lambda x: column_ordering[x])
    confusion = confusion[columns]
    return confusion


def no_false_negatives(confusion):
    return confusion[[x for x in confusion if x != "none"]]


def no_false_positives(confusion):
    return confusion.loc[[x for x in confusion.index if x != "none"]]


def compute_fne(confusion):
    confusion = no_false_positives(confusion)
    fne = (
        confusion["none"].sum() / confusion.sum().sum()
        if "none" in list(confusion)
        else 0
    )
    return fne


def compute_fpe(confusion):
    confusion = no_false_negatives(confusion)
    fpe = (
        confusion.loc["none"].sum() / confusion.sum().sum()
        if "none" in confusion.index
        else 0
    )
    return fpe


def compute_ce(confusion):
    confusion = no_false_negatives(confusion)
    confusion = no_false_positives(confusion)
    # compute the maximum diagonal over all possible permutations
    i_s, j_s = scipy.optimize.linear_sum_assignment(np.array(-confusion))
    ce = 1 - sum(confusion.iloc[i, j] for i, j in zip(i_s, j_s)) / confusion.sum().sum()
    return ce


def errors_from_confusion_without_multi(confusion):
    return compute_fne(confusion), compute_fpe(confusion), compute_ce(confusion)


def errors_from_model(m, domain, stamps, samples=10_000):
    confusion = compute_confusion_matrix(
        m, domain, stamps, samples=samples, handle_multi=False
    )
    return errors_from_confusion_without_multi(confusion)


@permacache(
    "pixel_art/analysis/evaluate_motifs/errors_from_checkpoint_4",
    key_function=dict(domain=stable_hash, stamps=stable_hash),
)
def errors_from_checkpoint(model_name, checkpoint_key, domain, stamps, samples=10_000):
    print(model_name, checkpoint_key)
    m = load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()
    return errors_from_model(m, domain, stamps, samples=samples)


def errors_from_path_and_density(
    model_name, target_density, domain, stamps, samples=10_000
):
    checkpoint_key = f"{target_density:.6e}"
    assert os.path.exists(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key)
    ), model_name
    return errors_from_checkpoint(
        model_name, checkpoint_key, domain, stamps, samples=samples
    )


def errors_from_all(models, target_density, domain, stamps, samples=10_000):
    fne = {}
    fpr = {}
    cr = {}
    for name, (path, num_seeds) in models.items():
        fne[name], fpr[name], cr[name] = {}, {}, {}
        for seed in range(1, 1 + num_seeds):
            (
                fne[name][seed],
                fpr[name][seed],
                cr[name][seed],
            ) = errors_from_path_and_density(
                f"{path}_{seed}", target_density, domain, stamps, samples=samples
            )
    return pd.DataFrame(fne) * 100, pd.DataFrame(fpr) * 100, pd.DataFrame(cr) * 100


def display_confusion(confusion, per_row, size=6, ax=None, *, xticks_kwargs={}):
    if ax is None:
        plt.figure(
            figsize=(size, 0.4 * size * confusion.shape[0] / confusion.shape[1]),
            facecolor="white",
            dpi=200,
            tight_layout=True,
        )
        ax = plt.gca()
    per_row = per_row / per_row.sum()
    ax.imshow(confusion, cmap="gray", aspect="auto")
    ax.set_yticks(
        np.arange(len(confusion.index)),
        [f"{x} [{y:6.2%}]" for x, y in zip(confusion.index, per_row)],
    )
    ax.set_xticks(np.arange(len(confusion.columns)), confusion.columns, **xticks_kwargs)
    for (j, i), label in np.ndenumerate(confusion):
        if label != 0:
            ax.text(
                i,
                j,
                f"{label:.2%}" if label < 0.01 else f"{label:.1%}",
                ha="center",
                va="center",
                color="black" if label > 0.5 else "white",
                size=8,
            )
