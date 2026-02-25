import itertools
import string

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt

from pixel_art.analysis.evaluate_motifs import evaluate_motifs
from pixel_art.domain.stamp import digit_stamps


def render_single_motif_example(
    x,
    y,
    mot,
    *,
    ax,
    title_fn,
    marker_size=50,
    show_motifs=True,
    invert,
    stretch=None,
    motif_offset=(0, 0),
    backmap,
):
    i_s, j_s = np.where((mot != 0).any(0))
    mot = mot[:, i_s, j_s]
    unique = (mot != 0).sum(0) == 1
    ax.imshow(
        x,
        cmap="gray_r" if invert else "gray",
        alpha=0.25 if show_motifs else 1,
        aspect=x.shape[1] / x.shape[0] * stretch if stretch is not None else 1,
    )
    ax.set_title(title_fn(y))
    if not show_motifs:
        return
    import seaborn as sns

    if len(backmap) <= 20:
        colors = sns.color_palette("tab20", len(backmap))
    elif len(backmap) == 21:
        colors = sns.color_palette("tab20") + [(0, 0, 0)]
    else:
        colors = sns.color_palette("tab20", len(backmap))
    mot_names = (
        string.ascii_uppercase
        if mot.shape[0] <= 26
        else [f"#{i:02d}" for i in range(mot.shape[0])]
    )
    for i, c in enumerate(mot_names[: mot.shape[0]]):
        for is_unique in True, False:
            mask = (mot.argmax(0) == i) & (unique == is_unique)
            if sum(mask) == 0:
                continue
            ax.scatter(
                j_s[mask] + motif_offset[0],
                i_s[mask] + motif_offset[1],
                marker="." if is_unique else "*",
                label=c + ("" if is_unique else "*"),
                s=marker_size,
                color=colors[backmap[i]],
            )


def render_examples_from_data(
    xs,
    ys,
    mots,
    axs,
    *,
    title_fn,
    show_motifs,
    invert=True,
    stretch=None,
    motif_offset=(0, 0),
    side="bottom",
    **kwargs,
):
    handles, texts = [], []

    backmap = {x: i for i, x in enumerate(np.where(mots.any((0, 2, 3)))[0])}

    for i in range(len(axs)):
        render_single_motif_example(
            xs[i],
            ys[i],
            mots[i],
            ax=axs[i],
            title_fn=title_fn,
            marker_size=50,
            show_motifs=show_motifs,
            invert=invert,
            stretch=stretch,
            motif_offset=motif_offset,
            backmap=backmap,
        )
        h, t = axs[i].get_legend_handles_labels()
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        handles += h
        texts += t

    results = {}
    for text, handle in zip(texts, handles):
        results[text] = (text, handle)
    results = sorted(results.items())
    results = [(x, y) for x, (_, y) in results]
    if results:
        if side == "bottom":
            kwargs.update(dict(loc="lower center", bbox_to_anchor=(0.5, 0)))
        elif side == "right":
            kwargs.update(dict(loc="center right", bbox_to_anchor=(1, 0.5)))
        else:
            raise ValueError(f"Unknown side {side}")
        ncol = kwargs.get("ncol", 1)
        results = list(itertools.chain(*[results[i::ncol] for i in range(ncol)]))
        texts, handles = zip(*results)
        axs[-1].legend(
            handles,
            texts,
            prop={"size": 8},
            **kwargs,
        )

    return (mots != 0).sum((0, 2, 3))


def render_examples(m, domain, axs, *, show_motifs=True, **kwargs):
    xs, ys, mots, _ = evaluate_motifs(m, domain, digit_stamps(), samples=len(axs))
    return render_examples_from_data(
        xs, ys, mots, axs, show_motifs=show_motifs, **kwargs
    )


def render_examples_from_dataset(m, dataset, axs, *, show_motifs, **kwargs):
    xs, ys = zip(*[(dataset[i][0], dataset[i][1]) for i in range(4, 4 + len(axs))])
    with torch.no_grad():
        mots = (
            m.run_motifs_without_post_sparse(
                torch.tensor(np.array(xs, dtype=np.float32)).cuda()
            )
            .cpu()
            .numpy()
        )
    return render_examples_from_data(
        xs, ys, mots, axs, show_motifs=show_motifs, **kwargs
    )


def render_several_motif_examples(m, domain, *, grid_size=4, scale=1):
    _, axs = plt.subplots(
        1,
        grid_size,
        figsize=(scale * grid_size, scale * 1.2),
        dpi=400,
        tight_layout=True,
    )
    axs = axs.flatten()

    return render_examples(m, domain, axs, title_fn="".join, ncol=2)
