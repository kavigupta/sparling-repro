import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    # from the tutorial https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_vals = torch.cos(position * div_term)
        if d_model % 2 == 1:
            cos_vals = cos_vals[:, :-1]
        pe[:, 1::2] = cos_vals
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class BasicLSTM(nn.Module):
    def __init__(self, channels, bidirectional):
        super().__init__()
        output_channels = channels
        if bidirectional:
            assert output_channels % 2 == 0
            output_channels //= 2
        self.lstm = nn.LSTM(channels, output_channels, bidirectional=bidirectional)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class ResidualUnit(nn.Module):
    """
    Residual unit proposed in "Identity mappings in Deep Residual Networks"
    by He et al.
    """

    def __init__(self, *, l, w, ar, use_padding, dimension=2):
        super().__init__()
        self.normalize1 = batch_norm_classes[dimension](l)
        self.normalize2 = batch_norm_classes[dimension](l)
        self.act1 = self.act2 = nn.ReLU()

        padding = (ar * (w - 1)) // 2 if use_padding else 0

        self.conv1 = conv_classes[dimension](l, l, w, dilation=ar, padding=padding)
        self.conv2 = conv_classes[dimension](l, l, w, dilation=ar, padding=padding)

    def forward(self, input_node):
        bn1 = self.normalize1(input_node)
        act1 = self.act1(bn1)
        conv1 = self.conv1(act1)
        assert conv1.shape == act1.shape
        bn2 = self.normalize2(conv1)
        act2 = self.act2(bn2)
        conv2 = self.conv2(act2)
        assert conv2.shape == act2.shape
        output_node = conv2 + input_node
        return output_node


conv_classes = {1: nn.Conv1d, 2: nn.Conv2d}

batch_norm_classes = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d}
