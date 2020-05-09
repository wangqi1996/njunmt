# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.common_utils import register

ACTIVATION = dict()


def register_activation(name: str):
    return register(name, ACTIVATION)


@register_activation("relu")
class Relu(nn.ReLU):
    pass


@register_activation("gelu")
class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.erf(input / math.sqrt(2.0)))


@register_activation("swish")
class swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = x * F.sigmoid(x)
        return output


def build_activation(name: str):
    if name is None:
        return None
    elif name not in ACTIVATION:
        raise KeyError("Unknown scheduler name {0}. Do not use lr_scheduling.".format(name))
    else:
        return ACTIVATION[name]()
