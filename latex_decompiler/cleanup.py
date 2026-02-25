import datetime
import os
import time

import numpy as np

from latex_decompiler.monitoring import load_all

from .utils import get_sparsity_from_checkpoint, load_steps


def steps_to_keep(
    path, min_window=20_000, care_about_sparsity=True, max_density_to_keep=float("inf")
):
    steps = load_steps(f"model/{path}")
    # steps
    steps = np.array(steps)

    if len(steps) < 6:
        return steps, np.ones(len(steps), dtype=bool)

    if care_about_sparsity:
        densities = [1 - get_sparsity_from_checkpoint(path, step) for step in steps]
        densities = np.array(densities)

        # keep all steps before/after sparsity changes
        keep = (densities != [*densities[1:], -1]) | (
            densities != [-1, *densities[:-1]]
        )
        keep &= densities < max_density_to_keep
    else:
        densities = None
        keep = np.zeros(len(steps), dtype=bool)
    # keep all steps near the end
    keep |= steps > steps[-6]
    if care_about_sparsity:
        # do not keep bad checkpoints
        keep &= densities == densities
    kept = set(steps[keep].tolist())
    for i in range(keep.shape[0]):
        prev_kept_steps = [x for x in kept if steps[i] - min_window < x <= steps[i]]
        if prev_kept_steps:
            continue
        kept.add(steps[i])
        keep[i] = 1
    assert kept == set(steps[keep].tolist())
    return steps, keep


def remove_redundant_steps(path, dry_run=False, **kwargs):
    steps, keep = steps_to_keep(path, **kwargs)
    path = f"model/{path}/"
    for step, k in zip(steps, keep):
        to_remove = f"{path}/model/{step}"
        if not k:
            print(f"Removing {to_remove}")
            if not dry_run:
                os.remove(to_remove)
        else:
            if dry_run:
                print(f"Keeping  {to_remove}")


def remove_steps_from_all(
    models,
    min_window,
    max_density_to_keep,
    function_to_load,
    care_about_sparsity=True,
    loop=False,
    dry_run=False,
):
    while True:
        for p, max_seed, _ in models.values():
            for seed in range(1, 1 + max_seed):
                p_seed = f"{p}_{seed}"
                remove_redundant_steps(
                    p_seed,
                    min_window=min_window,
                    max_density_to_keep=max_density_to_keep,
                    care_about_sparsity=care_about_sparsity,
                    dry_run=dry_run,
                )
                load_all(p_seed, function_to_load)
        if not loop:
            break
        print(datetime.datetime.now())
        time.sleep(120)
