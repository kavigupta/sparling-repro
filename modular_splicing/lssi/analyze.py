import numpy as np


def topk(yps, ys, c):
    """
    Compute the top-k accuracy for the given results and labels
    """
    return ((yps > np.quantile(yps, 1 - (ys == c).mean())) & (ys == c)).sum() / (
        ys == c
    ).sum()
