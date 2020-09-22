# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class FoggyDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features,out_dim=6, stride=1):
        super(FoggyDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.out_dim = out_dim

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(in_channels=self.num_ch_enc[-1], out_channels=256, kernel_size=1)
        self.convs[("foggy", 0)] = nn.Conv2d(in_channels=256, out_channels=128,kernel_size= 3, stride=stride, padding=1)
        self.convs[("foggy", 1)] = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride =stride, padding=1)
        self.convs[("foggy", 2)] = nn.Conv2d(in_channels=128, out_channels=self.out_dim , kernel_size=1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = input_features[-1]

        cat_features = self.relu(self.convs["squeeze"](last_features))

        out = cat_features
        for i in range(3):
            out = self.convs[("foggy", i)](out)#不同尺度
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1,  self.out_dim)

        return out
