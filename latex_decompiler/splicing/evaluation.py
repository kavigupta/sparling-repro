import itertools
import os
from collections import defaultdict
from functools import lru_cache

import numpy as np
import scipy
import torch
import tqdm.auto as tqdm
from matplotlib import pyplot as plt
from more_itertools import chunked
from permacache import permacache, stable_hash

from latex_decompiler.remapping_pickle import load_with_remapping_pickle
from latex_decompiler.splicing.splicing_dataset import SplicingDataset
from latex_decompiler.splicing.splicing_model import SplicingLSSI
from latex_decompiler.train import TopKValidation
from modular_splicing.models.motif_models.psam_fixed_motif import PSAMMotifModel
from modular_splicing.motif_names import get_motif_names
from pixel_art.analysis.evaluate_motifs import (
    categorize_relationships,
    confusion_from_results,
    errors_from_confusion_without_multi,
)
from pixel_art.theme import blue, darken, orange

SELECTED_CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "selected_checkpoints",
)

batch_size = 16


def test_set_iterator(pbar=True, amount=None):
    dset = SplicingDataset(is_training=False, seed=1)
    if amount is None:
        amount = dset.index_to_chunk.shape[0]
    dset = itertools.islice(dset, amount)
    return tqdm.tqdm(dset, total=amount) if pbar else dset


@permacache(
    "latex_decompiler/splicing/evaluation/evaluate_e2e",
    key_function=dict(mod=stable_hash),
)
def evaluate_e2e(mod):
    assert not mod.training
    all_ys, all_yps = [], []
    for xys in chunked(test_set_iterator(), batch_size):
        x, y = [np.array(u) for u in zip(*xys)]
        with torch.no_grad():
            yp = mod.forward_test(
                torch.tensor(x, dtype=torch.float32).cuda(), max_length=None
            )
        all_ys.append(y)
        all_yps.append(yp)
    all_ys = np.concatenate(all_ys)
    all_yps = torch.cat(all_yps)

    return TopKValidation().compute_accuracy(all_ys, all_yps)[0]


@lru_cache(None)
def compute_all_ys():
    all_ys = []
    for xys in chunked(test_set_iterator(), batch_size):
        y = np.array([u[1] for u in xys])
        all_ys.append(y)
    all_ys = np.concatenate(all_ys)
    return all_ys


@permacache("latex_decompiler/splicing/evaluation/evaluate_e2e_control")
def evaluate_e2e_control(seed):
    rng = np.random.RandomState(seed)
    all_ys = compute_all_ys()
    all_yps = rng.randn(*all_ys.shape)
    all_yps = torch.tensor(all_yps).softmax(dim=-1)
    return TopKValidation().compute_accuracy(all_ys, all_yps)[0]


@permacache("latex_decompiler/splicing/evaluation/evaluate_e2e_from_checkpoint_2")
def evaluate_e2e_from_checkpoint(model_name, checkpoint_key):
    print(model_name, checkpoint_key)
    mod = load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()
    return evaluate_e2e(mod)


def gather_motifs(motifs_fn, sparsify=True, amount=None):
    motifs_all = []
    for xs in chunked((x for x, _ in test_set_iterator(amount=amount)), batch_size):
        xs = torch.tensor(np.array(xs), dtype=torch.float32).cuda()
        with torch.no_grad():
            motifs = motifs_fn(xs).cpu().numpy()
            assert motifs.shape[1] == 5000 * 3
            motifs = motifs[:, 5000:-5000]
        if sparsify:
            motifs_all += [scipy.sparse.csr_matrix(motif) for motif in motifs]
        else:
            motifs_all.append(motifs)
    if not sparsify:
        motifs_all = np.concatenate(motifs_all)
    return motifs_all


@permacache("latex_decompiler/splicing/evaluation/motifs_from_checkpoint_2")
def motifs_from_checkpoint(model_name, checkpoint_key, amount=None):
    mod = load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()

    def motif_fn(x):
        res = mod.run_motifs(x)
        assert res.shape[-1] in {80, 82}
        if res.shape[-1] == 82:
            res = res[..., 2:]
        return res

    return gather_motifs(motif_fn, amount=amount)


@permacache("latex_decompiler/splicing/evaluation/splice_sites")
def get_lssi(amount=None):
    spl_lssi = (
        SplicingLSSI(
            acceptor="model/splicepoint-model-acceptor-1",
            donor="model/splicepoint-donor2-2.sh",
            in_channels=4,
            out_channels=2,
        )
        .cuda()
        .eval()
    )
    return gather_motifs(spl_lssi.forward_as_motifs, amount=amount)


@permacache("latex_decompiler/splicing/evaluation/get_fixed_motifs")
def get_fixed_motifs(amount=None):
    fm = (
        PSAMMotifModel(
            dict(type="rbns"),
            input_size=4,
            channels=None,
            num_motifs=79,
            exclude_names=("3P", "5P"),
        )
        .eval()
        .cuda()
    )
    result = gather_motifs(fm, sparsify=False, amount=amount)
    # sparsify to 0.18e-2
    thresholds = np.quantile(result.reshape(-1, result.shape[-1]), 1 - 0.18e-2, axis=0)
    result = result - thresholds
    result[result < 0] = 0
    return [scipy.sparse.csr_matrix(motif) for motif in result]


@permacache("latex_decompiler/splicing/evaluation/full_fms")
def full_fms(amount=None):
    lssis = get_lssi(amount=amount)
    fms = get_fixed_motifs(amount=amount)
    return [scipy.sparse.hstack([lssi, fm]) for lssi, fm in zip(lssis, fms)]


def evaluate_motif_on_sample(fm, lm, motif_width=21):
    names = motif_names()
    assert motif_width % 2 == 1
    pad = motif_width // 2
    stamps = [
        dict(slices=(slice(max(loc - pad, 0), loc + pad + 1),), symbol=names[which])
        for loc, which in zip(*fm.nonzero())
    ]
    return categorize_relationships(
        lm.toarray().T,
        placed_stamps=stamps,
        handle_multi=False,
        motif_names=[f"#{i}" for i in range(100)],
    )


@permacache("latex_decompiler/splicing/evaluation/motif_names")
def motif_names():
    return ["3P", "5P"] + get_motif_names("rbns")


def compute_sample_relationships(lms, amount=1_000):
    fms = full_fms(amount=amount)
    conf = []
    for fm, lm in zip(tqdm.tqdm(fms[:amount]), lms):
        conf += evaluate_motif_on_sample(fm=fm, lm=lm)

    return conf


@permacache("latex_decompiler/splicing/evaluation/evaluate_motifs_from_checkpoint_2")
def evaluate_motifs_from_checkpoint(model_name, checkpoint_key, amount=1_000):
    lms = motifs_from_checkpoint(model_name, checkpoint_key, amount=amount)
    return compute_sample_relationships(lms, amount=amount)


@permacache(
    "latex_decompiler/splicing/evaluation/evaluate_motifs_randomized_from_checkpoint_2"
)
def evaluate_motifs_randomized_from_checkpoint(
    model_name, checkpoint_key, *, seed, amount=1_000
):
    lms = motifs_from_checkpoint(model_name, checkpoint_key, amount=amount)
    lms = randomize(lms[:amount], seed)
    return compute_sample_relationships(lms, amount=amount)


def randomize(lms, seed):
    lms_random = np.array([lm.toarray() for lm in lms])
    n_batch, n_seq, n_motifs = lms_random.shape
    lms_random = lms_random.reshape(-1, n_motifs)
    rng = np.random.RandomState(seed)
    for idx in tqdm.trange(n_motifs):
        rng.shuffle(lms_random[:, idx])
    lms_random = lms_random.reshape(n_batch, n_seq, n_motifs)
    return [scipy.sparse.csr_matrix(lm) for lm in lms_random]


@permacache("latex_decompiler/splicing/evaluation/results_for_4")
def results_for(model_name, checkpoint_key, *, num_controls=100, amount=1000):
    conf = evaluate_motifs_from_checkpoint(model_name, checkpoint_key, amount=amount)
    return dict(
        e2e=evaluate_e2e_from_checkpoint(model_name, checkpoint_key),
        motif_errors=all_motif_errors(conf),
        motif_errors_control=[
            all_motif_errors(
                evaluate_motifs_randomized_from_checkpoint(
                    model_name, checkpoint_key, seed=seed, amount=amount
                )
            )
            for seed in tqdm.trange(num_controls)
        ],
    )


def all_motif_errors(conf):
    return {
        "direct": motif_errors_dict(conf),
        "grouped": motif_errors_dict(group_motifs(conf)),
        "no_3p5p": motif_errors_dict(remove_3p5p(conf)),
    }


def motif_errors_dict(conf):
    fne, fpe, ce = errors_from_confusion_without_multi(confusion_from_results(conf))
    return dict(fne=fne, fpe=fpe, ce=ce)


def classify_motif_names():
    result = defaultdict(list)
    for name in motif_names():
        if name in {"3P", "5P"}:
            result[name].append(name)
        elif name.startswith("SR") or name.startswith("TRA2A"):
            result["SR"].append(name)
        elif name.startswith("HN"):
            result["HN"].append(name)
        else:
            result["none"].append(name)
    return dict(result.items())


def group_motifs(conf):
    name_to_class = {v: k for k, vs in classify_motif_names().items() for v in vs}
    name_to_class["none"] = "none"
    return [(lm, name_to_class[fm]) for lm, fm in conf]


def remove_3p5p(conf):
    return [(lm, "none" if fm in {"3P", "5P"} else fm) for lm, fm in conf]


def plot_result_for_motif_metric(ax, results, error_type, legend=False):
    condition_to_name = {
        "direct": "full",
        # "grouped": "grouped",
        "no_3p5p": "no 3P/5P",
    }
    handles = {}
    xticks = {}
    for i, (name, res) in enumerate(results.items()):
        for seed in range(len(res)):
            for j, condition in enumerate(condition_to_name):
                color = f"C{i}"
                x = i * 0.3 + j
                xticks[(len(results) - 1) / 2 * 0.3 + j] = condition_to_name[condition]
                handles[name, "actual"] = ax.scatter(
                    x,
                    100 * res[seed]["motif_errors"][condition][error_type],
                    color=color,
                    marker=".",
                )
                control_vals = [
                    100 * x[condition][error_type]
                    for x in results[name][seed]["motif_errors_control"]
                ]
                bootstrap = np.random.choice(
                    control_vals, (1000, len(control_vals))
                ).mean(axis=1)
                lo, hi = np.percentile(bootstrap, [2.5, 97.5])
                handles[name, "control"] = ax.errorbar(
                    x,
                    (lo + hi) / 2,
                    yerr=(hi - lo) / 2,
                    color=color,
                    capsize=5,
                )
    # create a legend, using handles
    if legend:
        ax.legend(
            [handles[name, "actual"] for name in results]
            + [handles[name, "control"] for name in results],
            [name for name in results] + [f"{name} [control]" for name in results],
        )
    xs = sorted(xticks.keys())
    ax.set_xticks(xs)

    ax.set_xticklabels([xticks[x] for x in xs])

    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)

    ax.set_ylim(0, 101)
    ax.set_title(error_type.upper())


def plot_end_to_end_error(ax, results):
    # results[model][seed]["e2e"]
    handles = {}
    for i, (name, res) in enumerate(results.items()):
        for seed in range(len(res)):
            handles[name] = ax.scatter(
                i,
                100 * (1 - res[seed]["e2e"]),
                color=f"C{i}",
                marker=".",
            )

        e2e_control = [100 - 100 * evaluate_e2e_control(i) for i in range(10)]

        bootstrap = np.random.choice(e2e_control, (1000, len(e2e_control))).mean(axis=1)
        lo, hi = np.percentile(bootstrap, [2.5, 97.5])
        ax.errorbar(
            i,
            (lo + hi) / 2,
            yerr=(hi - lo) / 2,
            color=f"C{i}",
            capsize=5,
        )
    # ax.legend(handles.values(), handles.keys())
    ax.set_ylim(0, 101)
    ax.set_title("E2EE")
    ax.set_xticks([])
    ax.set_xlim(-1.5, len(results) - 1 + 1.5)


def plot_all_errors(results):
    fig, axs = plt.subplots(1, 4, figsize=(10, 4), tight_layout=True, sharey=True)
    plot_result_for_motif_metric(axs[0], results, "fne", legend=True)
    plot_result_for_motif_metric(axs[1], results, "fpe")
    plot_result_for_motif_metric(axs[2], results, "ce")
    plot_end_to_end_error(axs[3], results)
    axs[0].set_ylabel("Error (%)")
    for ax in axs:
        # y grid only
        ax.grid(axis="y")
    axs[0].set_ylim(0, 101)
    plt.savefig("output/on-splicing.png", facecolor="white", dpi=200)
    plt.show()


def plot_errors(ax, result, condition):
    xs = np.arange(4)
    xticks = ["FNE", "FPE", "CE", "E2EE"]
    xticks_lower = [x.lower() for x in xticks]
    for i, x in enumerate(xs[:3]):
        #     ax.bar(i, 100 * result[condition][xticks_lower[i]], color=f"C{i}")
        for seed in range(len(result)):
            ax.scatter(
                x,
                100 * result[seed]["motif_errors"][condition][xticks_lower[i]],
                color=darken(blue),
                marker=".",
            )
        control_vals = [
            100 * x[condition][xticks_lower[i]]
            for x in result[seed]["motif_errors_control"]
        ]
        bootstrap = np.random.choice(control_vals, (1000, len(control_vals))).mean(
            axis=1
        )
        lo, hi = np.percentile(bootstrap, [2.5, 97.5])
        ax.errorbar(
            x,
            (lo + hi) / 2,
            yerr=(hi - lo) / 2,
            color=darken(orange),
            capsize=5,
            markeredgewidth=1.5,
        )
    # ax.bar(
    #     xs[-1], 100 * (1 - result[condition][xticks_lower[-1]]), color=f"C{len(xs) - 1}"
    # )
    for seed in range(len(result)):
        dot_handle = ax.scatter(
            xs[-1],
            100 * (1 - result[seed]["e2e"]),
            color=darken(blue),
            marker=".",
        )
    e2e_control = [100 - 100 * evaluate_e2e_control(i) for i in range(10)]
    bootstrap = np.random.choice(e2e_control, (1000, len(e2e_control))).mean(axis=1)
    lo, hi = np.percentile(bootstrap, [2.5, 97.5])
    error_handle = ax.errorbar(
        xs[-1],
        (lo + hi) / 2,
        yerr=(hi - lo) / 2,
        color=darken(orange),
        capsize=5,
        markeredgewidth=1.5,
    )
    ax.set_xticks(xs)

    ax.set_xticklabels(xticks)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Error (%)")
    ax.grid(axis="y")
    ax.legend([dot_handle, error_handle], ["Sparling", "Random Baseline"])
