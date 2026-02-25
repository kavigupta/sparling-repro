import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from latex_decompiler.dataset import DATA_TYPE_MAP
from latex_decompiler.latex_cfg import LATEX_CFG_SPECS
from latex_decompiler.remapping_pickle import load_with_remapping_pickle
from latex_decompiler.utils import construct
from pixel_art.analysis.audio_mnist_experiment import (
    ac_multi_speaker_noisy_domain_train,
    get_data,
)
from pixel_art.analysis.audio_mnist_experiment import models as audio_models
from pixel_art.analysis.audio_mnist_experiment import sparsity_bar as audio_sparsity_bar
from pixel_art.analysis.latex_experiment import models as latex_models
from pixel_art.analysis.latex_experiment import sparsity_bar as latex_sparsity_bar
from pixel_art.analysis.main_experiment import data_spec, load_sparsity_bars
from pixel_art.analysis.motif_example import (
    render_examples,
    render_examples_from_dataset,
)
from pixel_art.domain.domain import domain_types
from pixel_art.domain.stamp import digit_stamps

SELECTED_CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "selected_checkpoints",
)


def axes_for_all_domains(width, *, shrink_width):
    # Plot all the domains
    # Pixel art: one on top, one on bottom
    # LaTeX: one on top, one on bottom, but each is 3x as wide
    # Audio MNIST: side-by-side, but each is 2x as wide as it is tall
    # use a subplot grid

    pixel_art_width = 1
    pixel_art_height = 1

    pixel_art_total_width = pixel_art_width
    pixel_art_total_height = pixel_art_height * 2

    latex_width = 3
    latex_height = 1

    latex_total_width = latex_width
    latex_total_height = latex_height * 2

    audio_mnist_width = 1
    audio_mnist_height = 2

    audio_mnist_total_width = audio_mnist_width * 2
    audio_mnist_total_height = audio_mnist_height

    assert pixel_art_total_height == latex_total_height == audio_mnist_total_height

    total_width = pixel_art_total_width + latex_total_width + audio_mnist_total_width

    total_height = pixel_art_total_height

    grid_shape = (total_height, total_width)

    plt.figure(
        figsize=(width, width * total_height / total_width / shrink_width),
        tight_layout=True,
    )

    # Pixel art
    pixel_art_axs = [
        plt.subplot2grid(
            grid_shape,
            (0, 0),
            rowspan=pixel_art_height,
        ),
        plt.subplot2grid(
            grid_shape,
            (pixel_art_height, 0),
            rowspan=pixel_art_height,
        ),
    ]

    # LaTeX
    latex_axs = [
        plt.subplot2grid(
            grid_shape,
            (0, pixel_art_total_width),
            rowspan=latex_height,
            colspan=latex_width,
        ),
        plt.subplot2grid(
            grid_shape,
            (latex_height, pixel_art_total_width),
            rowspan=latex_height,
            colspan=latex_width,
        ),
    ]

    # Audio MNIST
    audio_mnist_axs = [
        plt.subplot2grid(
            grid_shape,
            (0, pixel_art_total_width + latex_total_width),
            rowspan=audio_mnist_height,
            colspan=audio_mnist_width,
        ),
        plt.subplot2grid(
            grid_shape,
            (0, pixel_art_total_width + latex_total_width + audio_mnist_width),
            rowspan=audio_mnist_height,
            colspan=audio_mnist_width,
        ),
    ]

    return pixel_art_axs, latex_axs, audio_mnist_axs


def render_pixel_art(axs, show_motifs):
    actual = construct(domain_types(), data_spec["domain_spec"])
    m = load_digit_circle_model()
    render_examples(m, actual, axs, show_motifs=show_motifs, title_fn="".join, ncol=2)


def load_pixel_art_dataset_and_model():
    actual = construct(DATA_TYPE_MAP, data_spec, seed=0)
    m = load_digit_circle_model()
    return actual, m


def load_digit_circle_model():
    model_name = "pae-7bb1_1"
    density = load_sparsity_bars().right_above_line.sparsity_bar
    checkpoint_key = f"{density:.6e}"
    return load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()


def render_latex(axs, show_motifs):
    dset_noise, m = load_latex_dataset_and_model()
    render_examples_from_dataset(
        m,
        dset_noise,
        axs,
        show_motifs=show_motifs,
        title_fn=lambda y: " ".join(
            t.name.replace("PAREN", "P")
            .replace("SUP", "U")
            .replace("SUB", "D")
            .replace("FRACMID", "/")
            .replace("FRAC", "F")
            for t in y
        )
        + "\n"
        + "".join(t.code for t in y),
        ncol=10,
        columnspacing=0.5,
        handletextpad=-0.5,
        invert=False,
    )


def load_latex_dataset_and_model():
    latex_dataset = dict(
        type="NoisyBinaryLaTeXDataset",
        latex_cfg=LATEX_CFG_SPECS["latex_cfg"]["cfg"],
        font="computer_modern",
        data_config=dict(
            minimal_length=1,
            maximal_length=LATEX_CFG_SPECS["latex_cfg"]["maximal_length"],
            dpi=200,
            w=360,
            h=120,
        ),
        noise_amount=0.25,
    )
    dset_noise = construct(DATA_TYPE_MAP, latex_dataset, seed=1)
    model_name = latex_models["QBNLaTeX-32c"][0] + "_1"
    density = latex_sparsity_bar(32).sparsity_bar
    checkpoint_key = f"{density:.6e}"
    m = load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()

    return dset_noise, m


def render_audio_mnist(axs, stretch, show_motifs):
    dset, m = load_audio_dataset_and_model()
    render_examples_from_dataset(
        m,
        dset,
        axs,
        show_motifs=show_motifs,
        title_fn=lambda y: "".join(t.code for t in y),
        ncol=1,
        columnspacing=0.5,
        handletextpad=-0.5,
        stretch=stretch,
        motif_offset=(32, 0),
        side="right",
    )


def load_audio_dataset_and_model():
    dset = get_data(**ac_multi_speaker_noisy_domain_train)
    model_name = audio_models["Audio-10c [ms]/noise=-10"][0] + "_1"
    density = audio_sparsity_bar(10).sparsity_bar
    checkpoint_key = f"{density:.6e}"
    m = load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()

    return dset, m


def render_domains(*, show_motifs):
    pixel_art_axs, latex_axs, audio_mnist_axs = axes_for_all_domains(
        11, shrink_width=0.8
    )
    render_pixel_art(pixel_art_axs, show_motifs=show_motifs)
    render_latex(latex_axs, show_motifs=show_motifs)
    render_audio_mnist(audio_mnist_axs, 2.3, show_motifs=show_motifs)


def load_domain_by_key(key):
    return {
        "digit_circle": load_pixel_art_dataset_and_model,
        "latex_ocr": load_latex_dataset_and_model,
        "audio_mnist_sequence": load_audio_dataset_and_model,
    }[key]()
