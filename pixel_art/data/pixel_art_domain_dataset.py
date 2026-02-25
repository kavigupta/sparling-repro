import numpy as np

from latex_decompiler.cfg import Token
from latex_decompiler.utils import compute_seed, construct
from pixel_art.data.single_character_dataset import load_stamps
from pixel_art.domain.domain import domain_types


class PixelArtDomainDataset:
    def __init__(self, domain_spec, stamps_spec, seed, looping=None):
        self.domain = construct(domain_types(), domain_spec)
        self.stamps = load_stamps(stamps_spec)
        self.seed = seed
        self.looping = looping

    def __getitem__(self, idx):
        image, symbols = self.domain.sample(
            np.random.RandomState(compute_seed(self.seed, idx, self.looping)),
            self.stamps,
        )
        return image.astype(np.float32), [Token.single_symbol(s) for s in symbols]

    @property
    def data_config(self):
        # padding
        return dict(maximal_length=self.domain.max_syms + 2)


class PixelArtDomainSingleDigitDataset(PixelArtDomainDataset):
    def __getitem__(self, idx):
        image, [symbol] = super().__getitem__(idx)
        return image, int(symbol.name)
