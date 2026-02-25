import numpy as np

from pixel_art.analysis.audio_mnist_experiment import (
    audio_mnist_data_spec_test,
    audio_mnist_data_spec_train,
)
from pixel_art.analysis.gather_evaluation import get_minimum_sparsity_by_domain
from pixel_art.analysis.latex_digit_dataset import latex_digits_spec, raw_latex_digits
from pixel_art.analysis.pixel_art_experiment import (
    single_digit_spec as pixel_art_single_digit_spec,
)
from pixel_art.analysis.single_digit_motif_results import all_direct_eval

pixel_art_dsets = {
    "train": dict(type="pixel_art_data", dset_spec=pixel_art_single_digit_spec, seed=-2)
}

latex_dsets = {
    "train": dict(
        type="latex_data",
        dset_spec=latex_digits_spec,
        seed=-2,
    )
}

audio_dsets = {
    k: dict(type="audio_data", dset_spec=domain, seed=-2)
    for k, domain in [
        ("train", audio_mnist_data_spec_train),
        ("test", audio_mnist_data_spec_test),
    ]
}


def compute_all_direct_eval_results():
    latex_digits = sorted(raw_latex_digits())
    min_sparsity = get_minimum_sparsity_by_domain()

    all_results = {}
    all_results["DigitCircle"] = all_direct_eval(
        ("pae-7bb1", 9),
        min_sparsity["DigitCircle"],
        ("pae-11bb1", 9),
        list(range(0, 1 + 1_000_000, 100_000))[1:],
        pixel_art_dsets,
    )
    all_results["LaTeX-OCR"] = all_direct_eval(
        ("ltx-2dc3", 9),
        min_sparsity["LaTeX-OCR"],
        ("ltx-5dc3", 9),
        list(range(0, 1 + 3_000_000, 500_000))[1:],
        latex_dsets,
        num_model_classes=32,
        num_real_classes=19,
    )
    all_results["LaTeX-OCR [without +()]"] = all_direct_eval(
        ("ltx-2dc3", 9),
        min_sparsity["LaTeX-OCR"],
        ("ltx-5dc3", 9),
        list(range(0, 1 + 3_000_000, 500_000))[1:],
        latex_dsets,
        num_model_classes=32,
        num_real_classes=19,
        mask=np.array([digit not in "()+" for digit in latex_digits]),
    )
    all_results["AudioMNISTSequence"] = all_direct_eval(
        ("aum-2ka1", 9),
        min_sparsity["AudioMNISTSequence"],
        ("aum-4ka1", 9),
        list(range(0, 1 + 3_000_000, 500_000))[1:],
        audio_dsets,
    )
    return all_results
