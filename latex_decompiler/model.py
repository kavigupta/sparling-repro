import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cached_property import cached_property

from latex_decompiler.evaluate import step_for_sparsity

from .cfg import END, START
from .generic_modules import BasicLSTM, PositionalEncoding, ResidualUnit, conv_classes
from sparling import NoSparsity, sparsity_types
from .splicing.splicing_model import (
    SplicingDownstream,
    SplicingDownstreamCorrected,
    SplicingLSSI,
    SplicingMotifModel,
)
from .utils import construct, load_model, run_on_dimensions


class ExtractorCNN(nn.Module):
    # based on https://github.com/harvardnlp/im2markup/blob/eb29756a32e2f96eeeb0d31b58a061722ee45dc9/src/model/cnn.lua
    def __init__(self, in_channels, out_channels):
        """
        Input: (N, Cin, H, W)
        Output: (N, Cout, H / 8, W / 8)
        """
        super().__init__()
        assert out_channels % 8 == 0
        s = out_channels // 8
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels, s, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(s, 2 * s, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * s, 4 * s, 3, padding=1),
            nn.BatchNorm2d(4 * s),
            nn.ReLU(),
            nn.Conv2d(4 * s, 4 * s, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(4 * s, 8 * s, 3, padding=1),
            nn.BatchNorm2d(8 * s),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(8 * s, 8 * s, 3, padding=1),
            nn.BatchNorm2d(8 * s),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x - 0.5
        return self.stack(x)

    def notify_sparsity(self, sparsity):
        pass


class CollapseMotifs(nn.Module):
    def __init__(self, num_motifs, channels, size_reduction):
        """
        Input: (N, Cin, H, W)
        Output: (N, Cout, H / size_reduction, W / size_reduction)
        """
        super().__init__()
        self.reembed = nn.Conv2d(num_motifs, channels, kernel_size=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(size_reduction, size_reduction))

    def forward(self, x):
        if isinstance(self.reembed, nn.Conv1d):
            x = F.conv2d(
                x,
                weight=self.reembed.weight,
                bias=self.reembed.bias,
                stride=self.reembed.stride,
                padding=self.reembed.padding,
                dilation=self.reembed.dilation,
                groups=self.reembed.groups,
            )
        else:
            x = self.reembed(x)
        x = self.pool(x)
        return x


class CollapseMotifs1d(nn.Module):
    def __init__(self, num_motifs, channels, size_reduction):
        """
        Input: (N, Cin, H, W)
        Output: (N, Cout, H / size_reduction, W / size_reduction)
        """
        super().__init__()
        self.reembed = nn.Conv1d(num_motifs, channels, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=size_reduction)

    def forward(self, x):
        x = self.reembed(x)
        x = self.pool(x)
        return x


class CollapseMotifsAudio(CollapseMotifs1d):
    def forward(self, x):
        # x : N, C, L, 1
        x = x.squeeze(3)
        # x : N, C, L
        x = super(CollapseMotifsAudio, self).forward(x)
        # x : N, C2, L2
        x = x.unsqueeze(3)
        # x : N, C2, L2, 1
        return x


class ThresholdForKL(nn.Module):
    def __init__(self, threshold=0.5, *, and_then_spec, **kwargs):
        super().__init__()
        self.threshold = threshold
        self.and_then = construct(post_sparse_types(), and_then_spec, **kwargs)

    def forward(self, x):
        x = torch.nn.functional.relu(x - self.threshold)
        return self.and_then(x)


class MotifModel(ABC):
    @abstractmethod
    def forward(self, x):
        pass


class ConvolutionalMotifModel(nn.Module):
    def __init__(
        self, *, in_channels, out_channels, num_motifs, cr, dimension=2, cr_each=1
    ):
        super().__init__()
        assert (
            cr % (2 * cr_each) == 0
        ), f"residual unit has a context radius of (2 * cr_each)=({2 * cr_each})"
        self.conv_in = conv_classes[dimension](in_channels, out_channels, kernel_size=1)
        self.conv_stack = nn.Sequential(
            *[
                ResidualUnit(
                    l=out_channels,
                    w=2 * cr_each + 1,
                    ar=1,
                    use_padding=True,
                    dimension=dimension,
                )
                for _ in range(cr // (2 * cr_each))
            ]
        )
        self.conv_out = conv_classes[dimension](out_channels, num_motifs, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_stack(x)
        x = self.conv_out(x)
        return x


class AudioConvolutionalMotifModel(ConvolutionalMotifModel):
    def __init__(self, in_channels, **kwargs):
        self.in_channels = in_channels
        ConvolutionalMotifModel.__init__(
            self, in_channels=in_channels, **kwargs, dimension=1
        )

    def forward(self, x):
        # x : N, 1, L, C
        _, one, _, channels = x.shape
        assert one == 1
        assert channels == self.in_channels

        # reshape to N, L, C
        x = x.squeeze(1)  # N, L, C
        x = x.permute(0, 2, 1)  # N, C, L
        x = super(AudioConvolutionalMotifModel, self).forward(x)
        # reshape to N, C, L, 1
        x = x.unsqueeze(3)
        return x

    def notify_sparsity(self, sparsity):
        pass


class CondensingConvolutionalMotifModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        num_motifs,
        width,
        pool,
        layers,
        dimension=2,
    ):
        """
        Use a residual unit with width w, then max pool with kernel size pool, repeat

        See ultimate_width for a derivation of the effective input size
        """
        super().__init__()

        self.conv_in = conv_classes[dimension](in_channels, out_channels, kernel_size=1)
        seq = []
        for _ in range(layers):
            seq.append(
                ResidualUnit(
                    l=out_channels, w=width, ar=1, use_padding=True, dimension=dimension
                )
            )
            if dimension == 1:
                seq.append(nn.MaxPool1d(kernel_size=(pool)))
            else:
                assert dimension == 2
                seq.append(nn.MaxPool2d(kernel_size=(pool, pool)))
        self.conv_stack = nn.Sequential(*seq)
        self.conv_out = conv_classes[dimension](out_channels, num_motifs, kernel_size=1)
        self.width = width
        self.pool = pool
        self.layers = layers

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_stack(x)
        x = self.conv_out(x)
        return x

    def ultimate_width(self):
        """
        Use a residual unit with width w, then max pool with kernel size pool, repeat

        letting re = 2 * (width - 1)

        For a single layer, considering an output as a patch of size s,
            we have that pre-pooling, this has size
                pool_r * s,
            and the effective part of the input that can effect this has size
                pool_r * s + re

        Thus, to get the overall effective input size, we have
            1
            pool * 1 + re
            pool * (pool * 1 + re) + re
                = pool^2 * 1 + pool * re + re
            pool * (pool^2 * 1 + pool * re + re) + re
                = pool^3 * 1 + pool^2 * re + pool * re + re
        or in general, with n layers

            pool^n + re * sum_{i=0}^{n-1} pool^i

            pool^n + (2 * width - 1) / (pool - 1) * (pool^n - 1)

            ~=

            pool^n * (1 + 2 * (width - 1) / (pool - 1))

        """

        def geom(pool, layers):
            if pool == 1:
                return layers
            return (1 - pool**layers) / (1 - pool)

        return self.pool**self.layers + (2 * (self.width - 1)) * geom(
            self.pool, self.layers
        )


class StandardMultiSizeMotifModel(MotifModel):
    pass


class ConvolutionalMotifModelMultipleSizes(
    ConvolutionalMotifModel, StandardMultiSizeMotifModel
):
    def __init__(
        self,
        *,
        cfg,
        sizes_per_character,
        dataset_configuration,
        num_motifs=None,
        **kwargs,
    ):
        ConvolutionalMotifModel.__init__(
            self,
            **kwargs,
            num_motifs=(
                sizes_per_character * len(cfg.all_symbols())
                if num_motifs is None
                else num_motifs
            ),
        )

    def notify_sparsity(self, sparsity):
        pass


class CondensingConvolutionalMotifModelMultipleSizes(
    CondensingConvolutionalMotifModel, StandardMultiSizeMotifModel
):
    def __init__(self, *, cfg, sizes_per_character, dataset_configuration, **kwargs):
        if "num_motifs" not in kwargs:
            kwargs["num_motifs"] = sizes_per_character * len(cfg.all_symbols())
        CondensingConvolutionalMotifModel.__init__(self, **kwargs)

    def notify_sparsity(self, sparsity):
        pass


def PretrainedModel(path_to_model, step, trainable, key=(), **kwargs):
    print("Ignoring extra arguments", kwargs)
    step_actual, underlying_model = load_model(path_to_model, step)
    for k in key:
        underlying_model = underlying_model[k]
    assert step_actual == step, f"{path_to_model} does not contain step {step}"
    if not trainable:
        for p in underlying_model.parameters():
            p.requires_grad = False
    return underlying_model


def positional_encoding_types():
    return dict(PositionalEncoding=PositionalEncoding)


def transformer_encoder_types():
    return dict(
        TransformerEncoder=lambda channels, nhead, layers: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=channels, nhead=nhead), layers
        )
    )


class RowLSTMTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        row_lstm_spec,
        positional_embedding_spec=dict(type="PositionalEncoding"),
        transformer_encoder_spec,
        channels,
    ):
        """
        Input: (N, C, H, W)
        Output: (L, N, C)
        """
        super().__init__()
        self.row_lstm = construct(
            dict(BasicLSTM=BasicLSTM), row_lstm_spec, channels=channels
        )
        self.column_positional_embedding = construct(
            positional_encoding_types(),
            positional_embedding_spec,
            d_model=channels,
        )
        self.transformer_encoder = construct(
            transformer_encoder_types(),
            transformer_encoder_spec,
            channels=channels,
        )

    def forward(self, x):
        x = run_on_dimensions(self.row_lstm, x, ([3], [0, 2], [1]))
        x = run_on_dimensions(self.column_positional_embedding, x, ([2], [0, 3], [1]))
        # x : (N, C, H, W)
        x = x.reshape(*x.shape[:2], -1).permute(2, 0, 1)
        # x : (H * W, N, C)
        for layer in self.transformer_encoder.layers:
            assert getattr(layer, "norm_first", False) == False
            layer.norm_first = False
            assert getattr(layer.self_attn, "batch_first", False) == False
            layer.self_attn.batch_first = False
        x = self.transformer_encoder(x)
        return x


class Transformer1D(nn.Module):
    def __init__(
        self,
        *,
        positional_embedding_spec=dict(type="PositionalEncoding"),
        transformer_encoder_spec,
        channels,
    ):
        super().__init__()
        self.positional_embedding = construct(
            positional_encoding_types(),
            positional_embedding_spec,
            d_model=channels,
        )
        self.transformer_encoder = construct(
            transformer_encoder_types(),
            transformer_encoder_spec,
            channels=channels,
        )

    def forward(self, x):
        # x : (N, C, L)
        x = x.permute(2, 0, 1)
        # x : (L, N, C)
        x = self.positional_embedding(x)
        x = self.transformer_encoder(x)
        return x


class TransformerAudio(Transformer1D):
    def forward(self, x):
        # x : (N, C, L, 1)
        x = x.squeeze(3)
        # x : (N, C, L)
        x = super(TransformerAudio, self).forward(x)
        # x : (L, N, C)
        return x


class CFGDecoder(nn.Module):
    def __init__(self, *, cfg, include_start_end=True):
        super().__init__()
        tokens = cfg.all_tokens()
        if include_start_end:
            tokens = [START, END] + tokens
        self.tokens = tokens
        self.token_name_to_index = {tok.name: i for i, tok in enumerate(self.tokens)}

    def embed_tokens(self, program):
        return torch.tensor(
            [self.token_name_to_index[tok.name] for tok in program],
            device=next(self.parameters()).device,
        )

    def embed_all(self, programs):
        length = max(len(x) for x in programs)
        masks = torch.tensor(
            [
                [False] * len(program) + [True] * (length - len(program))
                for program in programs
            ],
            device=next(self.parameters()).device,
        )
        programs = torch.stack(
            [
                self.embed_tokens(program + [END] * (length - len(program)))
                for program in programs
            ]
        ).transpose(0, 1)
        return programs, masks


class TransformerCFGDecoder(CFGDecoder):
    def __init__(
        self,
        *,
        channels,
        cfg,
        positional_embedding_spec=dict(type="PositionalEncoding"),
        transformer_decoder_spec,
    ):
        super().__init__(cfg=cfg)
        self.transformer_decoder = construct(
            dict(
                TransformerDecoder=lambda channels, nhead, layers: nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(d_model=channels, nhead=nhead), layers
                )
            ),
            transformer_decoder_spec,
            channels=channels,
        )
        self.output_positional_embedding = construct(
            dict(PositionalEncoding=PositionalEncoding),
            positional_embedding_spec,
            d_model=channels,
        )
        self.output_embedding = nn.Embedding(len(self.tokens), channels)
        self.output_layer = nn.Linear(channels, len(self.tokens))

    def forward_train(self, encodings, ys):
        ys, masks = self.embed_all([[START] + y + [END] for y in ys])
        yps = self.produce_yps_predictions(encodings, ys, masks)
        masks = masks.T
        # ys    = <s> a b c    </s> </s> </s>
        # masks = 1   1 1 1    1    0    0
        # yps   = a   b c </s> </s> </s> </s>
        masks, ys = masks[1:], ys[1:]
        yps = yps[:-1]
        # ys    = a b c    </s> </s> </s>
        # masks = 1 1 1    1    0    0
        # yps   = a b c    </s> </s> </s>
        yps, ys = yps[~masks], ys[~masks]
        return nn.CrossEntropyLoss()(yps, ys)

    def produce_yps_predictions(self, encodings, ys, masks):
        embed = self.output_embedding(ys)
        embed = self.output_positional_embedding(embed)
        for layer in self.transformer_decoder.layers:
            assert getattr(layer, "norm_first", False) == False
            layer.norm_first = False
            assert getattr(layer.self_attn, "batch_first", False) == False
            layer.self_attn.batch_first = False
            assert getattr(layer.multihead_attn, "batch_first", False) == False
            layer.multihead_attn.batch_first = False
        yps_embeddings = self.transformer_decoder(
            embed,
            encodings,
            tgt_key_padding_mask=masks,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                sz=embed.shape[0], device=embed.device
            ),
        )
        yps = self.output_layer(yps_embeddings)
        return yps

    def forward_test(self, encodings, max_length):
        results = [[START] for _ in range(encodings.shape[1])]
        for _ in range(max_length):
            ys, masks = self.embed_all(results)
            yps = self.produce_yps_predictions(encodings, ys, masks)[-1]
            [
                res.append(self.tokens[y.item()])
                for res, y in zip(results, yps.argmax(1))
            ]
            if all(
                result[-1] == self.token_name_to_index["</s>"] for result in results
            ):
                break
        return results


class ConvolutionalDecoder(CFGDecoder):
    def __init__(
        self,
        *,
        channels,
        cfg,
        convolutional_model_spec,
        pool_amount,
        num_stacks,
        num_outputs,
        do_pool_padding=False,
    ):
        super().__init__(cfg=cfg, include_start_end=False)

        self.num_outputs = num_outputs
        self.convolutional_stacks = nn.ModuleList(
            [
                construct(
                    dict(ConvolutionalMotifModel=ConvolutionalMotifModel),
                    convolutional_model_spec,
                    in_channels=channels,
                    out_channels=channels,
                    num_motifs=channels,
                )
                for _ in range(num_stacks)
            ]
        )
        self.pool = nn.MaxPool2d(
            kernel_size=(pool_amount, pool_amount),
            padding=(pool_amount // 2) * do_pool_padding,
        )

        self.final_encoding = nn.Linear(channels, num_outputs * len(self.tokens))

    def forward(self, x):
        for stack in self.convolutional_stacks:
            x = stack(x)
            x = self.pool(x)
        x, _ = x.max(2)
        x, _ = x.max(2)
        x = self.final_encoding(x)
        x = x.view(x.shape[0], -1, len(self.tokens))
        return x

    def forward_train(self, encodings, ys):
        assert all(len(y) == self.num_outputs for y in ys), str((ys, self.num_outputs))
        ys, _ = self.embed_all(ys)
        ys = ys.T
        yps = self(encodings)
        return nn.CrossEntropyLoss()(yps.reshape(-1, yps.shape[-1]), ys.reshape(-1))

    def forward_test(self, encodings, max_length):
        yps = self(encodings)
        yps = yps.max(-1)[1]
        yps = yps.cpu().numpy()
        yps_tokens = [[START] + [self.tokens[i] for i in yp] + [END] for yp in yps]
        return yps_tokens


class LaTeXPredictor(nn.Module):
    def __init__(
        self,
        *,
        channels,
        motifs_spec,
        sparsity_spec,
        post_sparse_spec=dict(type="Identity"),
        encoder_spec,
        decoder_spec,
        in_channels=1,
        add_axis=True,
    ):
        super().__init__()

        self.motifs = construct(
            motif_model_types(),
            motifs_spec,
            in_channels=in_channels,
            out_channels=channels,
        )

        self.sparsity = construct(
            sparsity_types(),
            sparsity_spec,
            channels=channels,
        )

        self.post_sparse = construct(
            post_sparse_types(),
            post_sparse_spec,
            channels=channels,
        )

        self.encode = construct(
            dict(
                RowLSTMTransformerEncoder=RowLSTMTransformerEncoder,
                Transformer1D=Transformer1D,
                TransformerAudio=TransformerAudio,
                Identity=lambda channels: nn.Identity(),
            ),
            encoder_spec,
            channels=channels,
        )

        self.decode = construct(
            dict(
                TransformerCFGDecoder=TransformerCFGDecoder,
                ConvolutionalDecoder=ConvolutionalDecoder,
                SplicingDownstream=SplicingDownstream,
                SplicingDownstreamCorrected=SplicingDownstreamCorrected,
            ),
            decoder_spec,
            channels=channels,
        )

        self.add_axis = add_axis

    def run_motifs_without_post_sparse(
        self, xs, manipulate_motifs=lambda x: x, **kwargs
    ):
        # default is to add axis, as in the constructor
        if getattr(self, "add_axis", True):
            xs = xs[:, None]
        xs = self.motifs(xs)
        xs = self.sparsity(xs, **kwargs)
        xs = manipulate_motifs(xs)
        return xs

    def run_motifs_full(self, xs, manipulate_motifs=lambda x: x):
        mot_res = self.run_motifs_without_post_sparse(
            xs, manipulate_motifs=manipulate_motifs
        )
        if isinstance(mot_res, dict):
            mot = mot_res.pop("motifs_for_loss")
            xs = mot_res.pop("result")
        else:
            mot = xs = mot_res
        if hasattr(self, "post_sparse"):
            xs = self.post_sparse(xs)
        return mot, xs

    def run_motifs(self, xs):
        _, xs = self.run_motifs_full(xs)
        return xs

    def produce_encoding(self, xs, manipulate_motifs=lambda x: x):
        mot, xs = self.run_motifs_full(xs, manipulate_motifs=manipulate_motifs)
        xs = self.encode(xs)
        return mot, xs

    def forward_train(self, xs, ys, motif_losses=[]):
        mot, xs = self.produce_encoding(xs)
        motif_loss = sum([loss(mot) for loss in motif_losses])
        return self.decode.forward_train(xs, ys) + motif_loss

    def forward_test(self, xs, max_length, manipulate_motifs=lambda x: x):
        _, xs = self.produce_encoding(xs, manipulate_motifs=manipulate_motifs)
        return self.decode.forward_test(xs, max_length)

    @property
    def sparsity_value(self):
        return self.sparsity.sparsity

    @sparsity_value.setter
    def sparsity_value(self, value):
        self.sparsity.sparsity = value
        self.motifs.notify_sparsity(value)


class LaTeXPredictorJustMotifsModelStub(nn.Module):
    def __init__(self, *, channels, motifs_spec):
        super().__init__()

        self.motifs = construct(
            motif_model_types(),
            motifs_spec,
            in_channels=1,
            out_channels=channels,
        )

    def run_motifs_without_post_sparse(self, xs, disable_relu=False):
        xs = xs[:, None]
        xs = self.motifs(xs)
        return xs


def LoadWithFrozenMotifsAndNoBottleneck(model_path, at_density):
    assert os.path.exists(model_path), model_path
    assert os.path.dirname(model_path) == "model", model_path
    model_name = os.path.relpath(model_path, "model")

    step = step_for_sparsity(model_name, target_density=at_density)
    assert step is not None
    mod = load_model(model_path, step)[1]["model"]

    mod.sparsity = NoSparsity(starting_sparsity=0, channels=mod.sparsity.channels)
    for param in mod.motifs.parameters():
        param.requires_grad = False

    return mod


def full_model_types():
    return dict(
        LaTeXPredictor=LaTeXPredictor,
        LoadWithFrozenMotifsAndNoBottleneck=LoadWithFrozenMotifsAndNoBottleneck,
        LaTeXPredictorJustMotifsModelStub=LaTeXPredictorJustMotifsModelStub,
    )


def motif_model_types():
    # Lazy imports to break circular dependency: adjusted_model and parallel_models
    # both import from this module (model.py).
    from .parallel_models import ParallelMotifModels

    return dict(
        ExtractorCNN=ExtractorCNN,
        ConvolutionalMotifModelMultipleSizes=ConvolutionalMotifModelMultipleSizes,
        AudioConvolutionalMotifModel=AudioConvolutionalMotifModel,
        CondensingConvolutionalMotifModelMultipleSizes=CondensingConvolutionalMotifModelMultipleSizes,
        PretrainedModel=PretrainedModel,
        SplicingMotifModel=SplicingMotifModel,
        SplicingLSSI=SplicingLSSI,
        ParallelMotifModels=ParallelMotifModels,
    )


def post_sparse_types():
    return dict(
        CollapseMotifs=CollapseMotifs,
        CollapseMotifsAudio=CollapseMotifsAudio,
        CollapseMotifs1d=CollapseMotifs1d,
        Identity=lambda channels: nn.Identity(),
        ThresholdForKL=ThresholdForKL,
    )
