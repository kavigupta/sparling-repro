import unittest

import torch
from permacache import stable_hash

from latex_decompiler.dataset import DATA_TYPE_MAP
from latex_decompiler.experiments import e2e_dataset_1
from latex_decompiler.model import CondensingConvolutionalMotifModel
from latex_decompiler.utils import construct


class TestCondensingConv(unittest.TestCase):
    def ultimate_width(self, width, pool, layers):
        model = CondensingConvolutionalMotifModel(
            in_channels=25,
            out_channels=25,
            num_motifs=25,
            width=width,
            pool=pool,
            layers=layers,
            dimension=2,
        ).eval()
        # input with gradient
        x = torch.randn(1, 25, 1024, 1024, requires_grad=True)

        y = model(x)

        y = y**2
        y = y.sum((0, 1))

        y[y.shape[0] // 2, y.shape[1] // 2].backward()

        # locations of non-zero gradients
        grad = (x.grad != 0).any(0).any(0)
        nonzero = torch.nonzero(grad)
        return model, (nonzero.max(0).values - nonzero.min(0).values + 1).float().mean()

    def assert_ultimate_width(self, width, pool, layers):
        model, actual = self.ultimate_width(width, pool, layers)
        expected = model.ultimate_width()
        print(actual)
        print(expected)
        self.assertTrue((actual - expected).abs() < 2)

    def test_single_layer(self):
        self.assert_ultimate_width(3, 3, 1)
        self.assert_ultimate_width(5, 3, 1)
        self.assert_ultimate_width(7, 3, 1)
        self.assert_ultimate_width(3, 5, 1)

    def test_multi_layer(self):
        self.assert_ultimate_width(3, 3, 3)
        self.assert_ultimate_width(5, 3, 3)
        self.assert_ultimate_width(7, 3, 3)
        self.assert_ultimate_width(3, 5, 3)
        self.assert_ultimate_width(3, 3, 5)

    def test_no_pool(self):
        self.assert_ultimate_width(3, 1, 3)
        self.assert_ultimate_width(5, 1, 3)
        self.assert_ultimate_width(7, 1, 3)
        self.assert_ultimate_width(3, 1, 5)
        self.assert_ultimate_width(3, 1, 7)
