import matplotlib.pyplot as plt
import numpy as np
import tqdm.auto as tqdm

from latex_decompiler.evaluate import accuracies_at_density
from pixel_art.analysis.entropy_per_nonzero_activation import (
    compute_for_multiple_rounding_modes,
)
from pixel_art.analysis.evaluate_motifs import errors_from_all
from pixel_art.utils.bootstrap import bootstrap_mean

from .main_experiment import data_spec


def errors_by_density(
    models_to_evaluate_multi_density,
    sparsity_bars,
    actual,
    stamps,
    *,
    accuracy_metric,
    max_num_above_line
):
    densities = np.array(densities_to_plot(sparsity_bars))
    err_topline = topline_errors_each(
        models_to_evaluate_multi_density,
        sparsity_bars,
        accuracy_metric=accuracy_metric,
        max_num_above_line=max_num_above_line,
    )
    fne_all, fpe_all, ce_all = motif_errors_each(
        models_to_evaluate_multi_density,
        actual,
        stamps,
        sparsity_bars,
        max_num_above_line=max_num_above_line,
    )
    del fne_all  # no need to plot this
    del fpe_all  # no need to plot this
    errors_all = dict(
        E2EE=err_topline,
        # FNE=fne_all,
        # FPE=fpe_all,
        CE=ce_all,
    )
    return densities, errors_all


def hbern(delta):
    return -(delta * np.log(delta) + (1 - delta) * np.log(1 - delta)) / np.log(2)


def ent(delta, eta):
    return 10 * (hbern(delta) + delta * eta)


def plot_errors_by_entropy(densities, errors_all, eta):
    ents = ent(densities, eta)
    _, axs = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True, sharey=True, dpi=200)
    for model_key, ax in zip(["MT", "ST"], axs):
        for error_name, error in errors_all.items():
            low, high = zip(*[bootstrap_mean(x[model_key]) for x in error])
            ax.plot(ents, [x[model_key].mean() for x in error], label=error_name)
            ax.fill_between(ents, low, high, alpha=0.2)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Entropy/pixel [b]")
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_title(model_key)
        ax.grid()
    axs[0].set_ylabel("Error [%]")
    axs[1].legend()


def densities_to_plot(sparsity_bars, max_num_above_line):
    densities = [sparsity_bars.right_above_line.sparsity_bar]
    for _ in range(max_num_above_line):
        densities.append(densities[-1] / 0.75)
    return densities


def topline_errors_each(model, sparsity_bars, *, accuracy_metric, max_num_above_line):
    densities = densities_to_plot(sparsity_bars, max_num_above_line)
    return [
        100
        - accuracies_at_density(
            model, data_spec, density, accuracy_metric=accuracy_metric
        ).T
        for density in tqdm.tqdm(densities)
    ]


def motif_errors_each(
    models_for_error_calc, domain, stamps, sparsity_bars, *, max_num_above_line
):
    densities = densities_to_plot(sparsity_bars, max_num_above_line)
    fne_all, fpe_all, ce_all = zip(
        *[
            errors_from_all(
                models_for_error_calc,
                dens,
                domain,
                stamps,
            )
            for dens in tqdm.tqdm(densities)
        ]
    )
    return fne_all, fpe_all, ce_all


def compute_results_for_binned_rounding_modes(
    models_for_error_calc,
    sparsity_bars,
    num_bins_values,
    *,
    max_num_above_line,
    **kwargs
):
    densities = densities_to_plot(sparsity_bars, max_num_above_line)
    return [
        compute_for_multiple_rounding_modes(
            models_for_error_calc, num_bins_values, dens, **kwargs
        )
        for dens in tqdm.tqdm(densities)
    ]


def main():
    from pixel_art.analysis.main_experiment import load_sparsity_bars
    from pixel_art.analysis.pixel_art_experiment import models

    sparsity_bars = load_sparsity_bars()
    compute_results_for_binned_rounding_modes(
        models,
        sparsity_bars,
        [2, 4, 8, 16],
        max_num_above_line=10,
    )


if __name__ == "__main__":
    main()
