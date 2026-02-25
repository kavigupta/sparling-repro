import os

import editdistance
import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm
from permacache import drop_if, drop_if_equal, permacache, stable_hash

from .dataset import DATA_TYPE_MAP
from .remapping_pickle import load_with_remapping_pickle
from .utils import (
    construct,
    get_accuracy_from_checkpoint,
    get_sparsity_from_checkpoint,
    load_model,
    load_steps,
    run_batched_fn,
    strip_start_and_end_tokens,
)

SELECTED_CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "selected_checkpoints",
)


@permacache(
    "latex_decompiler/evaluate/evaluate_e2e_model",
    key_function=dict(model=stable_hash, pbar=None),
)
def evaluate_e2e_model(*, model, data_spec, pbar, batch_size=10, amount):
    dset = construct(DATA_TYPE_MAP, data_spec)

    ys, yps = run_batched_fn(
        lambda xs: model.forward_test(
            torch.tensor(np.array(xs)).float().cuda(),
            dset.data_config["maximal_length"],
        ),
        dset,
        amount,
        batch_size,
        pbar=pbar,
    )
    yps = [strip_start_and_end_tokens(yp) for yp in yps]
    return dict(ys=ys, yps=yps)


def dist(x, y):
    x = [x.name for x in x]
    y = [x.name for x in y]
    return editdistance.eval(x, y) / max(len(x), len(y))


def accuracy_e2e_model(*, amount, accuracy_metric, **kwargs):
    result = evaluate_e2e_model(amount=amount, **kwargs)
    assert len(result["ys"]) == len(result["yps"]) and len(result["yps"]) >= amount
    return compute_accuracy_from_sequences(
        result["ys"][:amount], result["yps"][:amount], accuracy_metric=accuracy_metric
    )


def compute_accuracy_from_sequences(ys, yps, *, accuracy_metric):
    if accuracy_metric == "exact":
        return np.mean([yp == y for y, yp in zip(ys, yps)])
    elif accuracy_metric == "edit-dist":
        return np.mean([1 - dist(yp, y) for y, yp in zip(ys, yps)])
    else:
        raise ValueError(f"Unknown accuracy metric {accuracy_metric}")


def load_sparse_model(path, *, target_step):
    """
    sparsity should always be derived from the previous checkpoint
    sparsity at the checkpoint is reduced before saving
    slight design bug in training script

    In any case, we can compute this easily by looking at the previous checkpoint.
    Nothing else should have actually been updated, so we just need to set the sparsity manually
    """
    steps = sorted(load_steps(f"model/{path}"))
    step_idx = steps.index(target_step)
    # just use the current step if we end up with a step that's 0
    step_idx = step_idx - 1 if step_idx > 0 else 0
    true_sparsity = get_sparsity_from_checkpoint(path, steps[step_idx])

    m = load_model(f"model/{path}", target_step)[1]["model"]

    m.sparsity_value = true_sparsity
    return m


def load_sparse_model_for_sparsity(path, *, target_density, target="exact"):
    step = step_for_sparsity(path, target_density=target_density, target=target)
    if step is None:
        return None
    return load_sparse_model(path, target_step=step)


def step_for_sparsity(path, *, target_density, target="exact"):
    steps, densities = densities_for_model(path)
    if target == "exact":
        matching_sparsity = np.abs(densities - target_density) / target_density < 0.01
    elif target == "lower":
        matching_sparsity = densities <= target_density
    else:
        raise ValueError(f"target should be one of 'exact' or 'lower', got {target}")
    matching_idxs = np.where(matching_sparsity)[0]
    if len(matching_idxs) == 0:
        return None
    if target == "exact":
        last_matching = matching_idxs.max()
    elif target == "lower":
        last_matching = matching_idxs.min()
    else:
        assert not "reachable"
    if last_matching == len(steps) - 1:
        return None
    step = int(steps[last_matching + 1])
    return step


def densities_for_model(path):
    steps = sorted(load_steps(f"model/{path}"))
    densities = 1 - np.array(
        [get_sparsity_from_checkpoint(f"{path}", step) for step in steps]
    )

    return steps, densities


@permacache(
    "latex_decompiler/evaluate/accuracy_at_checkpoint_2",
    key_function=dict(
        accuracy_metric=drop_if_equal("exact"), batch_size=drop_if(lambda x: True)
    ),
    multiprocess_safe=True,
)
def accuracy_at_checkpoint(
    model_name, checkpoint_key, data_spec, accuracy_metric, batch_size=20
):
    print(model_name, checkpoint_key)
    mod = load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()
    res = accuracy_e2e_model(
        model=mod,
        data_spec=dict(**data_spec, seed=-2),
        pbar=tqdm.tqdm,
        batch_size=batch_size,
        amount=10**4,
        accuracy_metric=accuracy_metric,
    )
    return res


def accuracies_at_density(models, data_spec, density, *, accuracy_metric):
    checkpoint_key = f"{density:.6e}"
    accs = {}
    for name, (p, max_seed) in models.items():
        accs[name] = {}
        for seed in range(1, 1 + max_seed):
            model_name = f"{p}_{seed}"
            if not os.path.exists(
                os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key)
            ):
                continue
            acc = accuracy_at_checkpoint(
                model_name,
                checkpoint_key,
                data_spec=data_spec,
                accuracy_metric=accuracy_metric,
            )
            accs[name][seed] = acc
    return pd.DataFrame(accs).T * 100
