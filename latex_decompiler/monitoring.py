import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from .utils import (
    get_accuracy_from_checkpoint,
    get_sparsity_from_checkpoint,
    hbern,
    load_steps,
)

versions = {"1e-5": "v3", "4e-6": "v3p3", "2e-6": "v3p6", "1e-6": "v4", "1e-7": "v5"}
colors = {
    "1e-5": "red",
    "4e-6": "orange",
    "2e-6": "green",
    "1e-6": "cyan",
    "1e-7": "blue",
}


def load_all(path, fn, load_key="model"):
    steps = load_steps(f"model/{path}/", key=load_key)
    results = []
    for step in steps:
        try:
            results.append(fn(path, step=step))
        except RuntimeError:
            results.append(np.nan)
    return steps, results


def plot_accuracies(ax, path, **kwargs):
    steps, accs = load_all(path, get_accuracy_from_checkpoint, load_key="val_result")
    ax.plot(steps, np.array(accs) * 100, **kwargs)
    ax.set_xlabel("Train samples")
    ax.set_ylabel("Accuracy [%]")
    ax.set_ylim(0, 100)
    ax.grid()


def setup_sparsity_axis(ax, which):
    getattr(ax, f"set_{which}scale")("log")
    getattr(ax, f"get_{which}axis")().set_major_formatter(
        mpl.ticker.StrMethodFormatter("{x:.3f}")
    )
    getattr(ax, f"set_{which}ticks")(
        [a * 10**b for a in [1, 2, 5] for b in range(-3, 2)]
    )


def compute_entropy(sparsity, num_motifs, bits_per_motif_act=0):
    """
    H(all channels) = num_motifs * H(individual activation)
        = num_motifs * (Hbern(sparsity) + sparsity * bits_per_motif_act)
    """
    hb = hbern(sparsity)
    return num_motifs * (hb + sparsity * bits_per_motif_act)


def plot_sparsities(
    ax, path, x_metric="steps", y_metric="sparsity", num_motifs=np.nan, **kwargs
):
    sparsity_derived_metrics = {"sparsity", "entropy"}

    steps, sparses = load_all(
        path, lambda *args, **kwargs: 1 - get_sparsity_from_checkpoint(*args, **kwargs)
    )
    steps, accs = load_all(path, get_accuracy_from_checkpoint)
    metrics = dict(
        sparsity=dict(
            values=np.array(sparses) * 100,
            label="Sparsity [%]",
            axis_setup=setup_sparsity_axis,
        ),
        entropy=dict(
            values=compute_entropy(np.array(sparses), num_motifs),
            label="Entropy [b/pixel]",
            axis_setup=setup_sparsity_axis,
        ),
        accuracy=dict(values=np.array(accs) * 100, label="Accuracy [%]"),
        steps=dict(values=steps, label="Train samples"),
    )
    if {x_metric, y_metric} & sparsity_derived_metrics:
        mask = np.array(sparses)[:-1] != sparses[1:]
        for metric in metrics:
            vals = np.array(metrics[metric]["values"])
            if metric in sparsity_derived_metrics:
                vals = vals[:-1]
            else:
                vals = vals[1:]
            vals = vals[mask]
            metrics[metric]["values"] = vals
    ax.plot(metrics[x_metric]["values"], metrics[y_metric]["values"], **kwargs)
    for which, metric in zip(["x", "y"], [x_metric, y_metric]):
        getattr(ax, f"set_{which}label")(metrics[metric]["label"])
        if "axis_setup" in metrics[metric]:
            metrics[metric]["axis_setup"](ax, which)
    ax.grid()


def plot_all_curves(models, multicolor=False):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(dpi=200, facecolor="white", figsize=(10, 10))
    for i, name in enumerate(models):
        p, max_seed, num_motifs = models[name]
        for seed in range(1, 1 + max_seed):
            plot_sparsities(
                plt.gca(),
                f"{p}_{seed}",
                label=name if seed == 1 else None,
                x_metric="accuracy",
                y_metric="entropy",
                marker=".",
                num_motifs=num_motifs,
                color=colors[i] if not multicolor else None,
                linestyle=["-", "--", "-.", ":", " "][(seed - 1) % 5],
            )
    plt.legend()
    plt.grid()
