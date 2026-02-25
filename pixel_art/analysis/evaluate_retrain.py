import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from latex_decompiler.evaluate import SELECTED_CHECKPOINTS_DIR, accuracy_at_checkpoint


def collect_retrain_results_for_step(
    model_path_prefix, num_seeds, step, data_spec, *, quiet, batch_size=100
):
    for_step = collect_retrain_results(
        model_path_prefix,
        num_seeds,
        data_spec,
        quiet=quiet,
        step_filter=lambda s: s == step,
        batch_size=batch_size,
    )
    assert step in for_step, model_path_prefix
    return pd.Series(for_step[step])


def collect_retrain_results(
    model_path_prefix, num_seeds, data_spec, *, quiet, step_filter, batch_size=100
):
    for_step = {}
    for i in range(1, 1 + num_seeds):
        model_name = f"{model_path_prefix}_{i}"
        checkpoint_dir = os.path.join(SELECTED_CHECKPOINTS_DIR, model_name)
        if not os.path.isdir(checkpoint_dir):
            continue
        for key in sorted(os.listdir(checkpoint_dir)):
            if not key.startswith("step_"):
                continue
            step = int(key[len("step_") :])
            if not step_filter(step):
                continue
            for_step[step] = for_step.get(step, {})
            a = accuracy_at_checkpoint(
                model_name,
                key,
                data_spec,
                "edit-dist",
                batch_size=batch_size,
            )
            if not quiet:
                print(i, step, 1 - a)
            for_step[step][i] = 100 * a
    return for_step
