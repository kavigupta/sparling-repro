from collections import defaultdict

import numpy as np
import tqdm.auto as tqdm
from permacache import drop_if_equal, permacache, stable_hash

from latex_decompiler.dataset import DATA_TYPE_MAP
from latex_decompiler.latex_cfg import LATEX_CFG_SPECS
from latex_decompiler.localize_latex_characters import compute_stamps
from latex_decompiler.utils import construct

latex_digits_spec = dict(
    type="SingleDigitNoisyBinaryLaTeXDataset",
    latex_cfg=None,
    font=None,
    data_config=None,
    noise_amount=0.25,
)


def pack_elements(elements, size):
    result = np.ones((len(elements), size, size))
    for i in range(len(elements)):
        h, w = elements[i].shape
        pad_h, pad_w = (size - h) // 2, (size - w) // 2
        result[i, pad_h : pad_h + h, pad_w : pad_w + w] = elements[i]
    return result


@permacache(
    "pixel_art/analysis/latex_digit_dataset/latex_digits",
    key_function=dict(num_samples=drop_if_equal(10_000)),
)
def raw_latex_digits(*, size=40, num_samples=10_000):
    spec = LATEX_CFG_SPECS["latex_cfg"]

    data_spec = dict(
        type="LaTeXDataset",
        latex_cfg=spec["cfg"],
        font="computer_modern",
        data_config=dict(
            minimal_length=1,
            maximal_length=spec["maximal_length"],
            dpi=200,
            w=360,
            h=120,
        ),
    )
    data = construct(DATA_TYPE_MAP, data_spec, seed=-10)

    elements = {}
    counts = defaultdict(int)
    characters = {}
    for i in tqdm.trange(num_samples):
        x, y = data[i]
        stamps = compute_stamps(y, data_spec["font"], data_spec["data_config"], pad=2)
        for stamp in stamps:
            if stamp["symbol"] == "FRACBAR":
                continue
            char = x[stamp["slices"]]
            h = stable_hash(char)
            if h in elements:
                assert characters[h] == stamp["symbol"]
            characters[h] = stamp["symbol"]
            elements[h] = char
            counts[h] += 1

    which_symbols = sorted(set(characters.values()))

    elements_by_symbol = {}
    for symbol in which_symbols:
        elements_by_symbol[symbol] = pack_elements(
            [
                elements[h]
                for h, sym in characters.items()
                if sym == symbol and counts[h] > 1
            ],
            size,
        )
    return elements_by_symbol


@permacache("pixel_art/analysis/latex_digits_dataset/digits_data_2")
def digits_data(dset_spec, seed, amount=10**4):
    from .single_digit_motif_results import compute_digits_data

    return compute_digits_data(dset_spec, seed, amount)
