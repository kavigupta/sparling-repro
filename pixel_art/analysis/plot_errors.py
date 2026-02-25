import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pixel_art.analysis.evaluate_motifs import errors_from_all
from pixel_art.analysis.gather_evaluation import get_minimum_sparsity_by_domain
from pixel_art.theme import color_by_column, color_ce, color_fpe, darken
from pixel_art.utils.bootstrap import bootstrap_mean


def plot_bars(xs, table, label, ax, width=0.4, *, color):
    ax.bar(
        xs,
        table.mean(),
        width=width,
        label={"e2e_edit": "e2e"}.get(label, label).upper(),
        color=color,
    )
    low, high = np.array(table.apply(bootstrap_mean, axis=0))
    ax.scatter(
        np.repeat(xs[None], table.shape[0], axis=0).flatten(),
        np.array(table).flatten(),
        color="black",
        marker=".",
    )
    ax.errorbar(
        xs,
        (low + high) / 2,
        (high - low) / 2,
        linestyle=" ",
        color="black",
        capsize=5,
    )


def plot_both_bars(fne, fpe, cr, ax, title):
    del fne  # do not need to plot this
    xs = np.arange(cr.shape[1])
    plot_bars(xs - 0.2, fpe, label="FPE", ax=ax, color=color_fpe)
    plot_bars(xs + 0.2, cr, label="CE", ax=ax, color=color_ce)
    ax.grid(axis="y")
    ax.set_xticks(xs, list(cr), rotation=60)
    ax.set_title(title)
    # ax.set_yscale()


def plot_errors(models_for_error_calc, domain, stamps, *, right_above_line):
    at_1p1 = errors_from_all(
        models_for_error_calc,
        right_above_line,
        domain,
        stamps,
    )
    at_1p5 = errors_from_all(
        models_for_error_calc,
        right_above_line / 0.75,
        domain,
        stamps,
    )
    _, axs = plt.subplots(1, 2, figsize=(6, 4), tight_layout=True, sharey=True, dpi=200)
    plot_both_bars(*at_1p1, axs[0], "At 1.1x theoretical minimum")
    plot_both_bars(*at_1p5, axs[1], "At 1.5x theoretical minimum")
    axs[0].set_ylabel("Error [%]")
    axs[1].legend()
    return at_1p1


def plot_errors_for_domain(
    ax, domain_results, columns, *, rotation=0, ylabel="Error [%]"
):
    xvals = []
    xticks = []
    tables = []
    prev = 0
    steps_above_class = {
        0: "Sparling",
        # 1: " 1 above min",
        "not-sparse": "Non-Sparse",
        "sparse-retrained": "Retrained",
    }
    for steps_above in domain_results:
        if steps_above not in steps_above_class:
            continue
        skip = False
        for model in domain_results[steps_above]:
            available_columns = set(domain_results[steps_above][model])
            if not available_columns.issuperset(columns):
                assert not (available_columns & set(columns))
                skip = True
        if skip:
            continue
        for model in domain_results[steps_above]:
            xvals.append(prev)
            name = steps_above_class[steps_above]
            if len(domain_results[steps_above]) > 1:
                name += f" ({model})"
            xticks.append(name)
            tables.append(domain_results[steps_above][model])
            prev += 1
        prev += 0.5
    xvals = np.array(xvals)

    width = 0.8 / len(columns)
    start = -0.8 / 2 + width / 2
    for i, col in enumerate(columns):
        plot_bars(
            xvals + start + width * i,
            pd.DataFrame([table[col] for table in tables]).T,
            col,
            ax,
            width,
            color=color_by_column[col],
        )
    if len(xticks) > 1:
        ax.set_xticks(xvals, xticks, rotation=rotation)
    else:
        ax.set_xticks([])
    ax.set_ylabel(ylabel)


def plot_all_errors(
    domains,
    all_results,
    *,
    columns,
    width_fixed=2,
    width_dynamic=2,
    height=3.5,
    **kwargs,
):
    widths = [width_fixed + width_dynamic * amount for _, _, amount in domains]
    _, axs = plt.subplots(
        1,
        len(domains),
        gridspec_kw={"width_ratios": widths},
        figsize=(sum(widths), height),
        tight_layout=True,
    )
    if len(domains) == 1:
        axs = [axs]
    for i, domain_name in enumerate(all_results):
        plot_errors_for_domain(axs[i], all_results[domain_name], columns, **kwargs)
        axs[i].set_title(domain_name)
    if len(columns) > 1:
        for ax in axs:
            ax.legend()


def plot_error_vs_sparsity_for_domain(ax, domain, result, columns):
    minimum_sparsity_by_domain = get_minimum_sparsity_by_domain()
    max_above = max(x for x in result if isinstance(x, int))

    idxs = np.arange(1 + max_above)
    for column in columns:
        sparsities = 100 * minimum_sparsity_by_domain[domain] / 0.75**idxs
        values = [result[i]["MT"][column] for i in idxs]
        means = np.array([np.mean(v) for v in values])
        lo, hi = np.array([bootstrap_mean(v) for v in values]).T
        ax.plot(
            sparsities,
            means,
            label=columns[column],
            color=darken(color_by_column[column]),
        )
        ax.fill_between(sparsities, lo, hi, alpha=0.25)
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.legend()
    ax.set_title(domain)
    ax.set_xlabel(r"$\delta$ [%]")
    ax.set_ylabel("Error [%]")
    ax.set_xticks(sparsities[::2])
    ax.get_xaxis().set_tick_params(which="minor", width=0)
    ax.get_xaxis().set_tick_params(which="minor", size=0)
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    # scientific notation at major ticks
    formatter = lambda x, pos: "{:.0g}".format(x)
    ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(formatter))


def plot_error_vs_sparsity(domain_to_result, columns):
    _, axs = plt.subplots(
        1,
        len(domain_to_result),
        figsize=(4 * len(domain_to_result), 2.75),
        dpi=200,
        tight_layout=True,
    )
    if len(domain_to_result) == 1:
        axs = [axs]
    for i, domain in enumerate(domain_to_result):
        plot_error_vs_sparsity_for_domain(
            axs[i], domain, domain_to_result[domain], columns
        )
