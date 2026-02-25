import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm
from permacache import permacache, stable_hash

from latex_decompiler.dataset import DATA_TYPE_MAP
from latex_decompiler.localize_latex_characters import compute_stamps
from latex_decompiler.remapping_pickle import load_with_remapping_pickle
from latex_decompiler.utils import construct, run_batched_fn_exact
from pixel_art.analysis.evaluate_motifs import (
    categorize_relationships,
    confusion_from_results,
)

SELECTED_CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "selected_checkpoints",
)


@permacache(
    "pixel_art/analysis/evaluate_latex_motifs/precisely_evaluate_latex_motifs_for_range_3",
    key_function=dict(mod=stable_hash, dset_spec=stable_hash),
)
def precisely_evaluate_latex_motifs_for_range(mod, dset_spec, *, start, end, pad):
    dset = construct(DATA_TYPE_MAP, dset_spec, seed=-2)
    _, mot_pr = run_batched_fn_exact(
        lambda x: mod.run_motifs_without_post_sparse(
            torch.tensor(np.array(x)).float().cuda()
        )
        .cpu()
        .numpy(),
        dset,
        start=start,
        end=end,
        batch_size=16,
        pbar=lambda x: x,
    )
    motif_names = [f"#{i}" for i in range(mot_pr[0].shape[0])]
    overall = []
    for i in range(start, end):
        result = categorize_relationships(
            mot_pr[i - start],
            placed_stamps=compute_stamps(
                dset[i][1], dset.font, dset.data_config, pad=pad
            ),
            handle_multi=False,
            motif_names=motif_names,
            include_stamp=True,
        )
        overall += [((i, stamp), m, r) for (stamp, m, r) in result]
    return overall


@permacache(
    "pixel_art/analysis/evaluate_latex_motifs/precisely_evaluate_latex_motifs_4",
    key_function=dict(mod=stable_hash, dset_spec=stable_hash),
)
def precisely_evaluate_latex_motifs(mod, dset_spec, *, num_samples, pad):
    overall = []
    chunk_size = 100
    for start in tqdm.trange(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        overall += precisely_evaluate_latex_motifs_for_range(
            mod, dset_spec, start=start, end=end, pad=pad
        )
    return overall


@permacache(
    "pixel_art/analysis/evaluate_latex_motifs/precisely_evaluate_latex_motifs_from_checkpoint_2",
    key_function=dict(dset_spec=stable_hash),
    multiprocess_safe=True,
)
def precisely_evaluate_latex_motifs_from_checkpoint(
    model_name, checkpoint_key, dset_spec, *, num_samples, pad
):
    assert checkpoint_key is not None
    print("evaluating latex motifs from checkpoint", model_name, checkpoint_key)
    mod = load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()
    return precisely_evaluate_latex_motifs(
        mod, dset_spec, num_samples=num_samples, pad=pad
    )


@permacache(
    "pixel_art/analysis/evaluate_latex_motifs/precisely_evaluate_latex_motifs_from_checkpoint_no_tags_2",
    key_function=dict(dset_spec=stable_hash),
    multiprocess_safe=True,
)
def precisely_evaluate_latex_motifs_from_checkpoint_no_tags(
    model_name, checkpoint_key, dset_spec, *, num_samples, pad
):
    return remove_tags(
        precisely_evaluate_latex_motifs_from_checkpoint(
            model_name, checkpoint_key, dset_spec, num_samples=num_samples, pad=pad
        )
    )


def remove_tags(tagged_overall):
    return [(m, r) for _, m, r in tagged_overall]


def motif_to_real_mapping(overall):
    by_motif = defaultdict(list)
    for mot, real in overall:
        if mot == "none" or real == "none":
            continue
        by_motif[mot].append(real)
    corresponding_real = {}
    for mot in by_motif:
        (most_common, _), *_ = Counter(by_motif[mot]).most_common()
        corresponding_real[mot] = most_common
    return corresponding_real


def confusion_error(overall):
    corresponding_real = motif_to_real_mapping(overall)
    return np.mean(
        [
            real != corresponding_real[mot]
            for mot, real in overall
            if not (mot == "none" or real == "none")
        ]
    )


def false_negative_error(overall):
    return np.mean([mot == "none" for mot, real in overall if not (real == "none")])


def false_positive_error(overall):
    return np.mean([real == "none" for mot, real in overall if not (mot == "none")])


def confused_sites(tagged_overall):
    corresponding_real = motif_to_real_mapping(remove_tags(tagged_overall))
    return [
        (i, m, r, corresponding_real[m])
        for i, m, r in tagged_overall
        if not (m == "none" or r == "none") and r != corresponding_real[m]
    ]


def unified_confusion_matrix(overall, rename="by_real"):
    confusion = confusion_from_results(overall)
    corresponding_real = motif_to_real_mapping(overall)
    confusion_collapsed = {}
    for real in {r for _, r in overall}:
        if real == "none":
            continue
        elements = [x for x in corresponding_real if corresponding_real[x] == real]
        if rename == "join" and not elements:
            continue
        if rename == "by_real":
            mot = "m" + real if real != "FRACBAR" else "m/"
        elif rename == "join":
            mot = "/".join(elements)
        else:
            raise ValueError(rename)
        confusion_collapsed[mot] = sum(confusion[x] for x in elements)
    confusion_collapsed["none"] = confusion["none"]
    confusion_collapsed = pd.DataFrame(confusion_collapsed)
    return confusion_collapsed
