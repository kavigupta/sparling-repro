from .kl_loss import KLLoss
from .l1_loss import L1Loss


def motif_loss_types():
    return dict(L1Loss=L1Loss, KLLoss=KLLoss)
