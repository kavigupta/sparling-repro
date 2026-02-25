import matplotlib.pyplot as plt

from pixel_art.analysis.audio_mnist_experiment import audio_mnist_confusion_matrix
from pixel_art.analysis.evaluate_motifs import display_confusion, realign_confusion
from pixel_art.analysis.latex_experiment import latex_confusion_matrix
from pixel_art.analysis.pixel_art_experiment import pixel_art_confusion_matrix


def plot_setup():
    # two row figure
    # first row has two plots
    # second row has one plot

    plt.figure(figsize=(15, 7), tight_layout=True)
    row_0, row_1 = 4, 6
    gridsize = (row_0 + row_1, 2)
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=1, rowspan=row_0)
    ax2 = plt.subplot2grid(gridsize, (0, 1), colspan=1, rowspan=row_0)
    ax3 = plt.subplot2grid(gridsize, (row_0, 0), colspan=2, rowspan=row_1)

    return ax1, ax2, ax3


_CONFUSION_MATRIX_FNS = dict(
    digit_circle=pixel_art_confusion_matrix,
    latex_ocr=latex_confusion_matrix,
    audio_mnist_sequence=audio_mnist_confusion_matrix,
)


def compute_confusion_matrices(num_samples=10_000, key=None):
    fns = {key: _CONFUSION_MATRIX_FNS[key]} if key else _CONFUSION_MATRIX_FNS
    return {k: realign_confusion(fn(num_samples=num_samples)) for k, fn in fns.items()}


def all_confusions():
    axs = plot_setup()
    confusions = compute_confusion_matrices()
    display_confusion(*confusions["digit_circle"], ax=axs[0])
    display_confusion(
        *confusions["latex_ocr"], ax=axs[2], xticks_kwargs=dict(rotation=30)
    )
    display_confusion(*confusions["audio_mnist_sequence"], ax=axs[1])
    axs[0].set_title("DigitCircle")
    axs[2].set_title("LaTeX-OCR")
    axs[1].set_title("AudioMNISTSequence")
