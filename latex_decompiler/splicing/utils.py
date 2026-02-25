import os

from modular_splicing.lssi.analyze import topk

SPLICEAI_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "spliceai_data",
)


def topk_both(yps, ys):
    assert len(ys.shape) == 2
    assert ys.shape + (3,) == yps.shape

    res = [topk(yps[:, :, c], ys, c) for c in (1, 2)]

    return sum(res) / len(res)
