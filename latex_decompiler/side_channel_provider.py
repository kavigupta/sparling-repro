from abc import ABC, abstractmethod

import numpy as np

from latex_decompiler.utils import construct


class StampBarcodeSCP(ABC):
    def __init__(
        self,
        blank_existing,
        placement_spec=dict(type="lower_right"),
        expansion_factor=1,
    ):
        self.blank_existing = blank_existing
        self.placement = construct(placement_specs(), placement_spec)
        self.expansion_factor = expansion_factor

    def __call__(self, *, seed, idx, x, y):
        x = x.copy()
        if self.blank_existing:
            x[:] = 1

        barcode = self.barcode(y)

        barcode = np.repeat(
            np.repeat(barcode, self.expansion_factor, axis=1),
            self.expansion_factor,
            axis=0,
        )

        x = self.placement(x=x, barcode=barcode)

        return x

    @abstractmethod
    def barcode(self, y):
        pass


class OneHotSideChannelProvider(StampBarcodeSCP):
    def __init__(self, toks, **kwargs):
        super().__init__(**kwargs)
        self.toks = toks

    def barcode(self, y):
        return 1 - np.eye(len(self.toks))[[self.toks.index(t) for t in y]].T


class BinaryBarcodeSideChannelProvider(StampBarcodeSCP):
    def __init__(self, toks, **kwargs):
        super().__init__(**kwargs)
        self.toks = toks
        self.n_bits = int(np.ceil(np.log(len(toks)) / np.log(2)))
        assert 2**self.n_bits >= len(toks)

    def barcode(self, y):
        idxs = [self.toks.index(tok) for tok in y]
        idxs = [("0" * self.n_bits + bin(x)[2:])[-self.n_bits :] for x in idxs]
        barcode = np.zeros((len(idxs) * 2 + 1, self.n_bits + 2)) + 0.5
        for i, b in enumerate(idxs):
            barcode[2 * i + 1, 1:-1] = list(map(int, b))
        barcode = barcode.T
        return barcode


def side_channel_providers():
    return dict(
        full_one_hot=OneHotSideChannelProvider,
        binary_barcode=BinaryBarcodeSideChannelProvider,
    )


def placement_specs():
    return dict(lower_right=PlaceLowerRight)


class PlaceLowerRight:
    def __call__(self, *, x, barcode):
        x[-barcode.shape[0] :, -barcode.shape[1] :] = np.minimum(
            barcode, x[-barcode.shape[0] :, -barcode.shape[1] :]
        )
        return x
