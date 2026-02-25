import os

import numpy as np
import torch
import torch.nn as nn

from modular_splicing.models.modules.lssi_in_model import (
    LSSI_MODEL_THRESH,
    BothLSSIModels,
)
from modular_splicing.models.modules.spliceai import SpliceAIModule
from modular_splicing.models.motif_models.learned_motif_model import LearnedMotifModel

from .utils import SPLICEAI_DATA_DIR


def SplicingMotifModel(
    in_channels,
    out_channels,
    motif_width,
    motif_feature_extractor_spec,
    motif_fc_layers,
):
    return LearnedMotifModel(
        input_size=in_channels,
        channels=out_channels,
        motif_width=motif_width,
        motif_feature_extractor_spec=motif_feature_extractor_spec,
        motif_fc_layers=motif_fc_layers,
        num_motifs=out_channels,
    )


class SplicingLSSI(nn.Module):
    def __init__(self, acceptor, donor, in_channels, out_channels):
        super().__init__()
        assert in_channels == 4 and out_channels == 2
        self.model = BothLSSIModels(
            os.path.join(SPLICEAI_DATA_DIR, acceptor),
            os.path.join(SPLICEAI_DATA_DIR, donor),
        )

    def forward(self, x):
        result, _ = self.model(x)
        return result

    def forward_as_motifs(self, x):
        spl = self.model.forward_just_splicepoints(x)
        spl = spl - LSSI_MODEL_THRESH
        spl = torch.nn.functional.relu(spl)
        return spl

    def notify_sparsity(self, sparsity):
        pass


class SplicingDownstream(nn.Module):
    def __init__(self, channels, window):
        super().__init__()
        self.model = SpliceAIModule(input_size=channels, window=window)
        self.logit_loss = nn.BCEWithLogitsLoss()

    def forward_test(self, x, max_length):
        return self.model(x)

    def forward_train(self, x, y):
        y = np.array(y, dtype=np.float32)
        y = torch.tensor(y, device=x.device)
        logits = self.forward_test(x, None)
        loss = self.logit_loss(logits, y)
        return loss


class SplicingDownstreamCorrected(nn.Module):
    def __init__(self, channels, window):
        super().__init__()
        self.model = SpliceAIModule(input_size=channels, window=window)
        self.logit_loss = nn.CrossEntropyLoss()

    def forward_test(self, x, max_length):
        return self.model(x).softmax(-1)

    def forward_train(self, x, y):
        y = np.array(y, dtype=np.float32)
        y = y.argmax(-1)
        y = torch.tensor(y, device=x.device)
        logits = self.model(x)
        loss = self.logit_loss(logits.reshape(-1, 3), y.flatten())
        return loss
