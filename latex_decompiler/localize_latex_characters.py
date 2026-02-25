import string

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from more_itertools import chunked
from permacache import permacache, stable_hash

from .dataset import image_for
from .render import hues


def color_latex_code(color, tok):
    code = tok.code
    if code in [*string.ascii_letters, *string.digits, "+"]:
        return rf"\color{{{color}}} {code}"
    if code == r"\left(":
        return rf"\cleft[{color}]("
    if code == r"\right)":
        return rf"\cright[{color}])"
    if code == r"\left[":
        return rf"\cleft[{color}]["
    if code == r"\right]":
        return rf"\cright[{color}]]"
    if tok.name == "FRAC(":
        return r"\color{" + color + r"} \frac{"
    if tok.name in {"FRACMID", "SUB(", "SUB)", "SUP(", "SUP)", "FRAC)"}:
        assert color == "black"
        return code
    raise RuntimeError(f"Unknown {tok}")


@permacache(
    "latex_decompiler/localize_latex_characters/character_bounding_boxes",
    key_function=dict(sample=stable_hash, data_config=stable_hash),
)
def character_bounding_boxes(sample, *, font, data_config):
    result = {}
    symbol_map = get_symbol_map(sample)
    for idx_chunk in chunked(sorted(symbol_map), len(hues)):
        _, bboxes = produce_bounding_boxes_for_indices(
            sample, idx_chunk, font=font, data_config=data_config
        )
        result.update(bboxes)
    return symbol_map, result


def produce_bounding_boxes_for_indices(sample, idx_chunk, *, font, data_config):
    hue_map = {i: hue for i, hue in zip(idx_chunk, hues)}
    colors = ["black"] * len(sample)
    for i in hue_map:
        colors[i] = f"color{hue_map[i]}"
    codes = [color_latex_code(color, tok) for color, tok in zip(colors, sample)]
    code = "".join(codes)
    img = image_for(code, font, data_config)
    hsv = matplotlib.colors.rgb_to_hsv(img / 255)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    hue = (hue.astype(np.float32) * 360 / 10).round().astype(np.int64) * 10 % 360

    bounding_boxes = {}
    for i, h in hue_map.items():
        mask = (hue == h) & (sat > 0.1)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        bounding_boxes[i] = (xmin, ymin, xmax, ymax)
    return img, bounding_boxes


def symbols(tok):
    if tok.name == "FRACMID":
        return ()
    if tok.name == "FRAC(":
        return ("FRACBAR",)
    return tok.rendered_symbols


def get_symbol_map(sample):
    symbol_map = {}
    for i, tok in enumerate(sample):
        syms = symbols(tok)
        if not syms:
            continue
        [x] = syms
        symbol_map[i] = x
    return symbol_map


def compute_stamps(sample, font, data_config, *, pad):
    symbol_map, result = character_bounding_boxes(
        sample, font=font, data_config=data_config
    )
    return [
        dict(
            symbol=symbol_map[i],
            slices=(
                slice(max(0, ymin - pad), ymax + pad + 1),
                slice(max(0, xmin - pad), xmax + pad + 1),
            ),
        )
        for i, (xmin, ymin, xmax, ymax) in result.items()
    ]


def plot_example(mod, dset, i, *, pad):
    img, sample = dset[i]
    symbols, result = character_bounding_boxes(
        sample, font=dset.font, data_config=dset.data_config
    )
    with torch.no_grad():
        mot = (
            mod.run_motifs_without_post_sparse(
                torch.tensor(np.array([img])).float().cuda()
            )
            .cpu()
            .numpy()[0]
        )
    _, ys, xs = np.where(mot)
    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap="gray")
    for xmin, ymin, xmax, ymax in result.values():
        plt.fill_between(
            [xmin - pad, xmax + pad],
            [ymin - pad, ymin - pad],
            [ymax + pad, ymax + pad],
            alpha=0.5,
        )
    plt.scatter(xs, ys)
