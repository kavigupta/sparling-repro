import itertools
from typing import List

import numpy as np
import torch
import tqdm.auto as tqdm
from permacache import drop_if_equal, permacache

from latex_decompiler.latex_cfg import latex_cfg
from latex_decompiler.utils import strip_start_and_end_tokens
from pixel_art.analysis.confusion_plot import compute_confusion_matrices
from pixel_art.display.display_domains import load_domain_by_key

maximal_radius = 5


def convert_back_to_idx(s):
    if s[0] == "#":
        return int(s[1:])
    assert len(s) == 1
    value = ord(s) - ord("A")
    assert 0 <= value < 10
    return value


def compute_columns(confusion, c):
    col = confusion.columns[np.argmax(confusion.loc[c])]
    assert confusion.loc[c, col] > 0.75
    result = []
    for x in col.split("/"):
        result.append(convert_back_to_idx(x))
    return result


def remove_no_maximal(mots, idxs):
    x, y = idxs[2], idxs[3]
    x2, y2 = x[:, None], y[:, None]
    dists = torch.maximum(torch.abs(x - x2), torch.abs(y - y2))
    within_r_mask = dists <= maximal_radius
    values = mots[list(idxs)].cpu()
    maximal, _ = (values * within_r_mask.float()).max(1)
    return idxs[:, values == maximal]


def replace(confusion, x, ys, on_motifs):
    cols_start = compute_columns(confusion, x)
    cols_end_each = [compute_columns(confusion, y)[0] for y in ys]

    def manipulate(mots):
        mots_copy = mots.clone()
        assert mots.shape[0] == 1
        idxs = torch.stack(torch.where(mots)).cpu()
        idxs = remove_no_maximal(mots, idxs)
        mask = (idxs[1] == torch.tensor(cols_start)[:, None]).any(0)
        idxs = idxs[:, mask]
        idxs = idxs[:, torch.argsort(idxs[3])]
        col_idx_to_use = [
            cols_end_each[i % len(cols_end_each)] for i in range(len(idxs[0]))
        ]
        print(col_idx_to_use)
        mots[
            idxs[0],
            col_idx_to_use,
            idxs[2],
            idxs[3],
        ] = mots[tuple(idxs)]
        mots[tuple(idxs)] = 0
        on_motifs(mots_copy.cpu().numpy(), mots.cpu().numpy(), idxs.cpu().numpy())
        return mots

    return manipulate


def run_perturbation(confusion, model, x, *, start_symbol, end_symbols):
    mots_res, mots_after_res, idxs_res = None, None, None

    def on_motifs(mots, mots_after, idxs):
        nonlocal mots_res, mots_after_res, idxs_res
        assert mots_res is None and mots_after_res is None and idxs_res is None
        mots_res = mots
        mots_after_res = mots_after
        idxs_res = idxs

    with torch.no_grad():
        [toks] = model.forward_test(
            torch.tensor(x)[None].float().cuda(),
            max_length=100,
            manipulate_motifs=replace(confusion, start_symbol, end_symbols, on_motifs),
        )
    toks = strip_start_and_end_tokens(toks)
    return toks, mots_res, mots_after_res, idxs_res


def run_symbol_perturbation(confusion, m, x, *, start_symbol, end_symbols):
    return run_perturbation(
        confusion, m, x, start_symbol=start_symbol, end_symbols=end_symbols
    )[0]


def compute_interchangables(key):
    if key == "latex_ocr":
        return [
            x
            for x in latex_cfg.all_symbols()
            if x.isdigit() or x.isalpha() and len(x) == 1
        ]
    assert key in ("digit_circle", "audio_mnist_sequence")
    return [str(i) for i in range(10)]


def valid_symbols_to_pick(candidates: List[str], y: List[object], *, key):
    if key in ("latex_ocr", "audio_mnist_sequence"):
        return candidates
    assert key == "digit_circle"
    y_names = [tok.name for tok in y]
    return [c for c in candidates if c not in y_names]


@permacache(
    "pixel_art/display/perturbations/run_symbol_perturbation_systematic_experiment_3",
    key_function=dict(confusion_num_samples=drop_if_equal(10_000)),
)
def symbol_perturbation_systematic_experiment(
    count=1000, *, key, confusion_num_samples=10_000
):
    dataset, m = load_domain_by_key(key)
    confusion, _ = compute_confusion_matrices(
        num_samples=confusion_num_samples, key=key
    )[key]
    interchangables = compute_interchangables(key)
    ys, y_perts, start_syms, end_syms = [], [], [], []
    pbar = tqdm.tqdm(total=count)
    for seed in itertools.count():
        x, y = dataset[seed]
        with torch.no_grad():
            [toks] = m.forward_test(
                torch.tensor(x)[None].float().cuda(),
                max_length=100,
            )
        y_pred = strip_start_and_end_tokens(toks)
        if y_pred != y:
            continue
        rng = np.random.default_rng(seed)
        relevant_symbols = [
            sym for sym in interchangables if sym in [tok.name for tok in y]
        ]
        if not relevant_symbols:
            continue
        start_symbol = rng.choice(relevant_symbols)
        end_symbol = rng.choice(
            valid_symbols_to_pick(
                [s for s in interchangables if s != start_symbol], y, key=key
            )
        )
        y_pert = run_symbol_perturbation(
            confusion, m, x, start_symbol=start_symbol, end_symbols=[end_symbol]
        )
        ys.append(y)
        y_perts.append(y_pert)
        start_syms.append(start_symbol)
        end_syms.append(end_symbol)
        pbar.update(1)
        if len(ys) >= count:
            break
    pbar.close()

    return ys, y_perts, start_syms, end_syms


def reorder(lst, *, key):
    if key in ("latex_ocr", "audio_mnist_sequence"):
        return lst
    assert key == "digit_circle"
    minimal_index = min(range(len(lst)), key=lambda i: int(lst[i]))
    return lst[minimal_index:] + lst[:minimal_index]


def compute_accuracy(ys, y_perts, start_syms, end_syms, *, key):
    results = []
    for y, y_pert, start_sym, end_sym in zip(ys, y_perts, start_syms, end_syms):
        y_names = [tok.name for tok in y]
        y_pert_names = [tok.name for tok in y_pert]
        assert start_sym in y_names
        is_acc = (
            reorder([x if x != start_sym else end_sym for x in y_names], key=key)
            == y_pert_names
        )
        results.append(float(is_acc))
    return np.array(results)


def compute_perturbation_accuracy(count=1000, *, key, confusion_num_samples=10_000):
    experiment = symbol_perturbation_systematic_experiment(
        count, key=key, confusion_num_samples=confusion_num_samples
    )
    return compute_accuracy(*experiment, key=key).mean()
