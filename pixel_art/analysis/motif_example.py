import itertools
import string

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from pixel_art.analysis.evaluate_motifs import evaluate_motifs
from pixel_art.domain.stamp import digit_stamps
from pixel_art.theme import color_series, darken


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
    R = 5
    c_s, i_s, j_s = np.where((mot != 0))
    mot_sparse = mot[c_s, i_s, j_s]

    points = []
    is_unique = []
    for idx in range(len(c_s)):
        i, j, c = i_s[idx], j_s[idx], c_s[idx]

        nearby = (np.abs(i_s - i) <= R) & (np.abs(j_s - j) <= R)
        if mot_sparse[nearby].max() != mot_sparse[idx]:
            continue
        points.append((i, j, c))
        is_unique.append(nearby.sum() == 1)

    ax.imshow(
        x,
        cmap="gray_r" if invert else "gray",
        alpha=0.5 if show_motifs else 1,
        aspect=x.shape[1] / x.shape[0] * stretch if stretch is not None else 1,
    )
    ax.set_title(title_fn(y))
    if not show_motifs:
        return
    n_channels = mot.shape[0]
    colors = [darken(color_series[i % len(color_series)]) for i in range(len(backmap))]
    mot_names = (
        string.ascii_uppercase
        if n_channels <= 26
        else [f"#{i:02d}" for i in range(n_channels)]
    )
    for (i, j, c), u in zip(points, is_unique):
        color = colors[backmap[c]]
        ax.plot(
            j + motif_offset[0], i + motif_offset[1],
            "." if u else "*",
            color=color, markersize=3,
        )
        name = mot_names[c]
        label = name if u else name + "*"
        ax.text(
            j + motif_offset[0], i + motif_offset[1], label,
            color=color,
            fontsize=8, fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5),
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
        axs[i].set_xticks([])
        axs[i].set_yticks([])

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
