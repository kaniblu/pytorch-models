"""
Pooling is the basic operation that maps a tensor of size [batch_size, k, dim]
to [batch_size, dim]. This package contains pooling implementations for when
k is variable.
"""

import torch

from torchmodels import utils
from torchmodels import common


class AbstractPooling(common.Module):

    def __init__(self, dim):
        super(AbstractPooling, self).__init__()
        self.dim = dim

    def forward(self, x, lens=None):
        raise NotImplementedError()


class MaxPooling(AbstractPooling):

    name = "max-pooling"

    def pool(self, x):
        return x.max(1)[0]

    def forward(self, x, lens=None):
        if lens is not None:
            mask = utils.mask(lens, x.size(1)).unsqueeze(-1)
            x = x.masked_fill(1 - mask, float("-inf"))
        return self.pool(x)


class SumPooling(AbstractPooling):

    name = "sum-pooling"

    def pool(self, x):
        return x.sum(1)

    def pool_dynamic(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        return self.pool(x * mask)

    def forward(self, x, lens=None):
        if lens is not None:
            mask = utils.mask(lens, x.size(1)).unsqueeze(-1)
            x *= mask.float()
        return self.pool(x)


class MeanPooling(AbstractPooling):

    name = "mean-pooling"

    def forward(self, x, lens=None):
        if lens is None:
            return x.mean(1)
        mask = utils.mask(lens, x.size(1)).unsqueeze(-1).float()
        return (x * mask).sum(1) / lens.unsqueeze(-1).float()


class LastPooling(AbstractPooling):

    name = "last-pooling"

    def forward(self, x, lens=None):
        if lens is None:
            return x[:, -1]
        return torch.index_select(x, 1, lens - 1)