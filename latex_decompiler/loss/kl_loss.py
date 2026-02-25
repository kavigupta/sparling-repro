import torch


class KLLoss:
    """
    KL loss for motif.
    """

    def __init__(self, weight, target):
        self.weight = weight
        self.target = target

    def __call__(self, motif):
        phat = motif.mean()
        p = self.target
        kl = p * torch.log(p / phat) + (1 - p) * torch.log((1 - p) / (1 - phat))
        return kl * self.weight
