import torch
import torch.nn as nn

from latex_decompiler.model import motif_model_types
from latex_decompiler.utils import construct


class ParallelMotifModels(nn.Module):
    def __init__(self, motif_model_specs, out_channels_each, in_channels, out_channels):
        super().__init__()
        assert len(motif_model_specs) == len(out_channels_each)
        assert sum(out_channels_each) == out_channels, (out_channels_each, out_channels)
        self.out_channels_each = out_channels_each
        self.motif_models = nn.ModuleList(
            [
                construct(
                    motif_model_types(),
                    motifs_spec,
                    in_channels=in_channels,
                    out_channels=out_channels_this,
                )
                for motifs_spec, out_channels_this in zip(
                    motif_model_specs, out_channels_each
                )
            ]
        )

    def forward(self, x):
        out = [motif_model(x) for motif_model in self.motif_models]
        out = torch.cat(out, dim=-1)
        return out

    def notify_sparsity(self, sparsity):
        for motif_model in self.motif_models:
            motif_model.notify_sparsity(sparsity)
