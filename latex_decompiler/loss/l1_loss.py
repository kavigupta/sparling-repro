import torch


class L1Loss:
    """
    L1 loss for motif.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, motif):
        return torch.mean(torch.abs(motif)) * self.weight
