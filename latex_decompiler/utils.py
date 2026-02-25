import hashlib
import os

import numpy as np
import torch
from permacache import permacache

from latex_decompiler.remapping_pickle import load_with_remapping_pickle

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT_PATH, "model")


def construct(type_map, spec, **default_kwargs):
    spec = dict(spec.items())
    type_name = spec.pop("type")
    default_kwargs.update(spec)
    return type_map[type_name](**default_kwargs)


def run_on_dimensions(fn, x, idx_selector):
    all_idxs = [idx for idxs in idx_selector for idx in idxs]
    assert sorted(all_idxs) == list(
        range(len(x.shape))
    ), "indices do not line up with input indices"
    original_shape = x.shape
    x = x.permute(all_idxs)
    nonflat_shape = list(x.shape)
    x = x.reshape(
        [np.prod([original_shape[idx] for idx in idxs]) for idxs in idx_selector]
    )
    flat_shape = list(x.shape)
    x = fn(x)
    idx = 0
    for dim, idxs in enumerate(idx_selector):
        if len(idxs) != 1:
            assert (
                x.shape[dim] == flat_shape[dim]
            ), "non-singleton dimensions can't be changed by dim"
        else:
            nonflat_shape[idx] = x.shape[dim]
        idx += len(idxs)
    x = x.reshape(*nonflat_shape)
    x = x.permute([all_idxs.index(i) for i in range(len(x.shape))])
    return x


def load_model(folder, step=None, key="model", step_filter=lambda x: True, **kwargs):
    def hook(m):
        if hasattr(m, "_load_hook"):
            m._load_hook()
        return m

    if not torch.cuda.is_available():
        kwargs.update(dict(map_location=torch.device("cpu")))

    if os.path.isfile(folder):
        return None, hook(load_with_remapping_pickle(folder, **kwargs))

    model_dir = os.path.join(folder, key)
    if not os.path.exists(model_dir):
        return None, None

    if step is None and os.listdir(model_dir):
        step = max(
            [step for step in os.listdir(model_dir) if step_filter(int(step))], key=int
        )

    path = os.path.join(model_dir, str(step))
    if not os.path.exists(path):
        return None, None

    return int(step), hook(load_with_remapping_pickle(path, **kwargs))


def load_steps(folder, key="model"):
    model_dir = os.path.join(folder, key)
    if not os.path.exists(model_dir):
        return []
    return sorted(int(x) for x in os.listdir(model_dir))


def save_model(model, folder, step, *, key="model"):
    path = os.path.join(folder, key, str(step))
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
    torch.save(model, path)


def compute_seed(seed, idx, looping):
    if looping is not None:
        idx %= looping
    return (
        int(
            hashlib.sha256(str((seed, idx)).encode("ascii")).hexdigest(),
            16,
        )
        % 2**32
    )


def run_batched_fn(fn, data_val, amount, batch_size, *, pbar=lambda x: x):
    all_ys, all_ys_pred = [], []
    for count in pbar(range(0, int(amount) + batch_size - 1, batch_size)):
        xs, ys = zip(*[data_val[i] for i in range(count, count + batch_size)])
        with torch.no_grad():
            ys_pred = fn(xs)
        all_ys.extend(ys)
        all_ys_pred.extend(ys_pred)
    return all_ys, all_ys_pred


def run_batched_fn_exact(fn, data_val, start, end, batch_size, *, pbar=lambda x: x):
    """
    Other function is slightly incorrect and gives too many values
    """
    all_ys, all_ys_pred = [], []
    for count in pbar(range(start, end, batch_size)):
        xs, ys = zip(*[data_val[i] for i in range(count, min(count + batch_size, end))])
        with torch.no_grad():
            ys_pred = fn(xs)
        all_ys.extend(ys)
        all_ys_pred.extend(ys_pred)
    return all_ys, all_ys_pred


@permacache("latex_decompiler/utils/get_accuracy_from_checkpoint")
def get_accuracy_from_checkpoint(path, step):
    print(path, step)
    _, x = load_model(os.path.join(MODEL_PATH, path), step=step, key="val_result")
    return x["val_fn_result"]["accuracy"]


@permacache(
    "latex_decompiler/utils/get_sparsity_from_checkpoint", key_function=dict(step=int)
)
def get_sparsity_from_checkpoint_cache(path, step):
    print(path, step)
    _, x = load_model(
        os.path.join(MODEL_PATH, path),
        step=step,
        key="model",
        map_location=torch.device("cpu"),
    )
    return x["model"].sparsity.sparsity


def get_sparsity_from_checkpoint(path, step):
    try:
        return get_sparsity_from_checkpoint_cache(path, step)
    except RuntimeError:
        return np.nan


def hbern(s):
    return -(np.log(s) * s + np.log(1 - s) * s) / np.log(2)


def strip_start_and_end_tokens(y):
    result = []
    assert y[0].name == "<s>"
    for x in y[1:]:
        if x.name == "</s>":
            break
        result.append(x)
    return result
