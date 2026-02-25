import functools
from types import SimpleNamespace

import attr
import numpy as np
import tqdm.auto as tqdm

from latex_decompiler.evaluate import accuracies_at_density
from latex_decompiler.monitoring import compute_entropy
from pixel_art.analysis.plot_errors import plot_bars
from pixel_art.domain.stamp import digit_stamps
from pixel_art.domain.stamp_circle import StampCircleDomain
from pixel_art.theme import color_e2e
from pixel_art.utils.bootstrap import bootstrap_mean


@attr.s
class SparsityBar:
    sparsity_bar = attr.ib()
    num_motifs = attr.ib(default=10)

    @classmethod
    def right_above_line(cls, spar, *args, **kwargs):
        right_above_line = 0.5 * 0.75 ** np.floor(np.log(spar / 0.5) / np.log(0.75))

        return cls(right_above_line, *args, **kwargs)

    @property
    def entropy_bar(self):
        return compute_entropy(self.sparsity_bar, num_motifs=self.num_motifs)


domain_kwargs = dict(
    size=100,
    min_radius=20,
    random_shift=3,
    max_syms=6,
    pre_noise=0.5,
    post_noise=0.05,
)

data_spec = dict(
    type="PixelArtDomainDataset",
    domain_spec=dict(type="StampCircleDomain", **domain_kwargs),
    stamps_spec=dict(type="digit_stamps"),
)


@functools.lru_cache(None)
def load_sparsity_bars():
    stamps = digit_stamps()
    domain_kwargs_wo_noise = domain_kwargs.copy()
    domain_kwargs_wo_noise["pre_noise"] = 0
    domain_kwargs_wo_noise["post_noise"] = 0
    idealized = StampCircleDomain(**domain_kwargs_wo_noise)
    idealized_samples = [
        idealized.sample(np.random.RandomState(i), stamps) for i in range(1000)
    ]
    pixels_per_char_wo_noise = np.mean([x.mean() for x, _ in idealized_samples]) / 10
    one_pixel_per_char = np.mean([len(y) / x.size for x, y in idealized_samples]) / 10

    return SimpleNamespace(
        pixels_per_char_wo_noise=SparsityBar(pixels_per_char_wo_noise),
        one_pixel_per_char=SparsityBar(one_pixel_per_char),
        right_above_line=SparsityBar.right_above_line(one_pixel_per_char),
    )


def plot_density(ax, models, density, columns_to_plot, *, accuracy_metric):
    sparsity_bars = load_sparsity_bars()

    df_full = 100 - accuracies_at_density(
        models, data_spec, density, accuracy_metric=accuracy_metric
    )
    df = df_full.loc[columns_to_plot]
    xs = np.arange(df.shape[0])

    plot_bars(xs, df.T, "", ax, color=color_e2e)
    ax.set_xticks(xs, df.index)
    ax.grid()
    ax.set_title(
        f"{density/sparsity_bars.one_pixel_per_char.sparsity_bar:.1f}x theoretical minimum"
    )
    return df_full
