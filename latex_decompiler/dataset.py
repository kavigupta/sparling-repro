import multiprocessing
import os
import pickle
import sqlite3
from functools import lru_cache

import attr
import numpy as np
import scipy.sparse
from permacache import permacache, stable_hash

from latex_decompiler.sequential_disk_file import SequentialCacheDiskFile
from latex_decompiler.side_channel_provider import side_channel_providers
from latex_decompiler.splicing.splicing_dataset import SplicingDataset
from pixel_art.audo_mnist.data.dataset import (
    AudioMNISTDomainDataset,
    AudioMNISTSingleDigitDomainDataset,
)
from pixel_art.data.pixel_art_domain_dataset import (
    PixelArtDomainDataset,
    PixelArtDomainSingleDigitDataset,
)
from pixel_art.data.single_character_dataset import PixelArtSingleCharacterDataset

from .render import character_representations, clipped_image, render
from .utils import compute_seed, construct


def padding(x, w, h):
    if x.shape[0] > h:
        extra = x.shape[0] - h
        left, right = extra // 2, extra - extra // 2
        x = x[left:-right]
    if x.shape[1] > w:
        extra = x.shape[1] - w
        left, right = extra // 2, extra - extra // 2
        x = x[:, left:-right]

    padding = h - x.shape[0]
    top, bot = padding // 2, padding - padding // 2
    padding = w - x.shape[1]
    left, right = padding // 2, padding - padding // 2

    x = np.pad(x, [(top, bot), (left, right), (0, 0)], constant_values=255)
    return x


def generate_several_datapoints_direct(latex_cfg, i_s, fonts, data_config):
    assert len(i_s) == len(fonts)
    if len(i_s) < 10:
        return [
            generate_single_datapoint_direct(latex_cfg, i, font, data_config)
            for i, font in zip(i_s, fonts)
        ]
    with multiprocessing.Pool(30) as p:
        return p.starmap(
            generate_single_datapoint_direct,
            [(latex_cfg, i, font, data_config) for i, font in zip(i_s, fonts)],
        )


def generate_single_datapoint_direct(latex_cfg, i, font, data_config):
    sample = latex_cfg.rejection_sample(
        seed=i,
        minimal_length=data_config["minimal_length"],
        maximal_length=data_config["maximal_length"],
    )
    x = image_for("".join(x.code for x in sample), font, data_config)[:, :, 0]
    res = (scipy.sparse.csr_matrix(255 - x), sample)
    return res


def image_for(code, font, data_config):
    im = render(code, font, dpi=data_config["dpi"])
    x = padding(np.array(im), data_config["w"], data_config["h"])
    return x


cache_dir = "cache"


@attr.s
class LaTeXDataset:
    latex_cfg = attr.ib()
    font = attr.ib()
    data_config = attr.ib()
    seed = attr.ib()
    looping = attr.ib(default=None)

    def __attrs_post_init__(self):
        key = stable_hash((self.latex_cfg, self.font, self.data_config, self.seed))
        cache_path = os.path.join(cache_dir, key)
        self._cache = SequentialCacheDiskFile(cache_path, self._compute)

    def _compute(self, idxs):
        return generate_several_datapoints_direct(
            self.latex_cfg,
            [compute_seed(self.seed, idx, self.looping) for idx in idxs],
            [self.font_for(idx) for idx in idxs],
            self.data_config,
        )

    def font_for(self, idx):
        if isinstance(self.font, list):
            return self.font[np.random.RandomState(idx).randint(len(self.font))]
        return self.font

    def __getitem__(self, idx):
        if self.looping is not None:
            idx %= self.looping
        x, y = self._cache[idx]
        x = 255 - x.toarray()
        return x / 255, y


class NoisyBinaryLaTeXDataset:
    def __init__(
        self,
        *,
        noise_amount,
        binarization_threshold=0.75,
        technique="set_false",
        **kwargs,
    ):
        self.data = LaTeXDataset(**kwargs)
        self.noise_amount = noise_amount
        self.binarization_threshold = binarization_threshold
        self.technique = technique

    def produce_noise(self, seed, idx, shape):
        noise_seed = int(stable_hash((seed, idx, "noise")), 16) % 2**32
        return np.random.RandomState(noise_seed).rand(*shape) < self.noise_amount

    def postprocess(self, x, idx):
        x = x > self.binarization_threshold
        if self.technique == "set_false":
            x = x & ~self.produce_noise(self.data.seed, idx, x.shape)
        elif self.technique == "flip":
            x = x ^ self.produce_noise(self.data.seed, idx, x.shape)
        else:
            raise ValueError(f"Unknown technique {self.technique}")
        return x

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = self.postprocess(x, idx)
        return x, y

    def font_for(self, idx):
        return self.data.font_for(idx)

    @property
    def font(self):
        return self.data.font

    @property
    def data_config(self):
        return self.data.data_config

    @property
    def latex_cfg(self):
        return self.data.latex_cfg


class SingleDigitNoisyBinaryLaTeXDataset:
    def __init__(self, *, num_samples=10_000, **kwargs):
        self.data = NoisyBinaryLaTeXDataset(**kwargs)
        # Lazy import to break circular dependency: latex_digit_dataset imports DATA_TYPE_MAP
        # from this module.
        from pixel_art.analysis.latex_digit_dataset import raw_latex_digits

        self.digits = raw_latex_digits(num_samples=num_samples)
        self.digit_to_idx = {k: i for i, k in enumerate(sorted(self.digits))}

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        digit = rng.choice(list(self.digits))
        y = self.digit_to_idx[digit]
        x = self.digits[digit]
        x = x[rng.randint(x.shape[0])]
        x = self.data.postprocess(x, idx)
        x = x.astype(np.float32)
        return x, y

    @property
    def font(self):
        return self.data.font

    @property
    def data_config(self):
        return self.data.data_config


class LaTeXDigitLevelDataset:
    def __init__(self, underlying_data_spec, **kwargs):
        self.underlying_data = construct(DATA_TYPE_MAP, underlying_data_spec, **kwargs)
        self.symbol_to_idx = {
            sym: idx
            for idx, sym in enumerate(
                sorted(self.underlying_data.latex_cfg.all_symbols())
            )
        }

    def __getitem__(self, idx):
        # Lazy import to break circular dependency: localize_latex_characters imports
        # image_for from this module.
        from latex_decompiler.localize_latex_characters import character_bounding_boxes

        img, samp = self.underlying_data[idx]
        syms, boxes = character_bounding_boxes(
            samp,
            font=self.underlying_data.font_for(idx),
            data_config=self.underlying_data.data_config,
        )
        return img, [
            dict(symbol=syms[k], box=boxes[k], symbol_id=self.symbol_to_idx[syms[k]])
            for k in sorted(boxes)
            if syms[k] in self.symbol_to_idx
        ]


@permacache("latex_decompiler/dataset/generate_single_character_datapoint_5")
def generate_single_character_datapoint(*, configuration, font, seed):
    creator = SingleCharacterDatumCreator(
        configuration=configuration, font=font, seed=seed
    )
    creator.create()
    return dict(image=creator.image, stamps=creator.stamps)


class SingleCharacterDatumCreator:
    def __init__(self, *, configuration, font, seed):
        self.configuration = configuration
        self.font = font
        self.stamps = []
        self._image = np.ones((self.s * 3, self.s * 3), dtype=np.float32)
        self.rng = np.random.RandomState(seed)

    @property
    def s(self):
        return self.configuration["image_size"]

    @property
    def image(self):
        return self._image[self.s : -self.s, self.s : -self.s]

    def stamp_centrally(self, symbol, latex):
        self.stamp(symbol, latex, self.s // 2, self.s // 2, target=True)

    def stamp_randomly(self, symbol, latex):
        while True:
            x, y = self.rng.choice(self.s, size=2)
            if (
                max(abs(x - self.s // 2), abs(y - self.s // 2))
                > self.configuration["central_protection"]
            ):
                break
        self.stamp(symbol, latex, x, y, target=False)

    def stamp(self, symbol, latex, x, y, **kwargs):
        dpi = self.rng.randint(
            self.configuration["min_dpi"], 1 + self.configuration["max_dpi"]
        )
        image = clipped_image(latex, self.font, dpi) / 255
        imw, imh = image.shape[1], image.shape[0]
        imx, imy = imw // 2, imh // 2
        xstart, ystart = x - imx, y - imy
        image_slice = slice(ystart + self.s, ystart + self.s + imh), slice(
            xstart + self.s, xstart + self.s + imw
        )
        self._image[image_slice] = np.min([self._image[image_slice], image], axis=0)
        self.stamps.append(
            dict(
                symbol=symbol,
                x=x,
                y=y,
                relative_size=dpi / self.configuration["max_dpi"],
                **kwargs,
            )
        )

    def create(self):
        central_character = self.random_character()
        self.stamp_centrally(
            central_character,
            self.rng.choice(character_representations(central_character)),
        )
        for _ in range(
            self.rng.choice(1 + self.configuration["max_number_characters"])
        ):
            symbol = self.random_character(central_character)
            self.stamp_randomly(
                symbol, self.rng.choice(character_representations(symbol))
            )

    def random_character(self, *excluded):
        return self.rng.choice(
            [x for x in self.configuration["cfg"].all_symbols() if x not in excluded]
        )


DATA_TYPE_MAP = dict(
    LaTeXDataset=LaTeXDataset,
    NoisyBinaryLaTeXDataset=NoisyBinaryLaTeXDataset,
    SingleDigitNoisyBinaryLaTeXDataset=SingleDigitNoisyBinaryLaTeXDataset,
    LaTeXDigitLevelDataset=LaTeXDigitLevelDataset,
    PixelArtSingleCharacterDataset=PixelArtSingleCharacterDataset,
    PixelArtDomainDataset=PixelArtDomainDataset,
    PixelArtDomainSingleDigitDataset=PixelArtDomainSingleDigitDataset,
    AudioMNISTDomainDataset=AudioMNISTDomainDataset,
    AudioMNISTSingleDigitDomainDataset=AudioMNISTSingleDigitDomainDataset,
    SplicingDataset=SplicingDataset,
)
