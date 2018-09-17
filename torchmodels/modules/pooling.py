"""
Pooling is the basic operation that maps a tensor of size [batch_size, k, dim]
to [batch_size, dim]
"""
import torch

from torchmodels import utils
from torchmodels import common


class BasePooling(common.Module):
    def __init__(self, dim):
        super(BasePooling, self).__init__()
        self.dim = dim

    def forward(self, x, lens=None):
        raise NotImplementedError()


class MaxPooling(BasePooling):
    name = "max-pool"

    def pool(self, x):
        return x.max(1)[0]

    def pool_dynamic(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        x = x * mask + (-mask + 1) * float("-inf")
        return self.pool(x)

    def forward(self, x, lens=None):
        max_len = x.size(1)
        if lens is not None:
            return self.pool_dynamic(x, utils.mask(lens, max_len))
        else:
            return self.pool(x)


class SumPooling(BasePooling):
    name = "sum-pool"

    def pool(self, x):
        return x.sum(1)

    def pool_dynamic(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        return self.pool(x * mask)

    def forward(self, x, lens=None):
        max_len = x.size(1)
        if lens is not None:
            return self.pool_dynamic(x, utils.mask(lens, max_len))
        else:
            return self.pool(x)


class MeanPooling(BasePooling):
    name = "mean-pool"

    def forward(self, x, lens=None):
        max_len = x.size(1)
        if lens is None:
            return x.mean(1)
        mask = utils.mask(lens, max_len).unsqueeze(-1).float()
        return (x * mask).sum(1) / lens.unsqueeze(-1).float()


class LastPooling(BasePooling):
    name = "last-pool"

    def forward(self, x, lens=None):
        if lens is None:
            return x[:, -1]
        return torch.index_select(x, 1, lens - 1)