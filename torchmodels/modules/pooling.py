from torchmodels import utils
from torchmodels import common


class AbstractPooling(common.Module):
    r"""An abstract class for pooling operations. A pooling operation
    takes variable number of vectors and produces a single vector of the same
    dimensions using some form of aggregation. Supports variable numbers of
    items for pooling.

    Shape:
        - Input:
          - vectors: :math: `(N, K, H)` FloatTensor
        - Output:
          - aggregated vector: :math: `(N, H)` FloatTensor

    Minimum Args:
        dim (int): Input and output feature dimensions

    Examples::

        >>> x = torch.randn(16, 8, 100)
        >>> # `FooPooling` is a subclass of `AbstractPooling`
        >>> pool = FooPooling(100)
        >>> pool.size()
        torch.Size([16, 100])
        >>> lens = torch.randint(4, 9, (16, ))
        # pool only from the first lens[i] items in each set of candidates x[i]
        >>> res = pool(x, lens)
        >>> res.size()
        torch.Size([16, 100])

    """

    def __init__(self, dim):
        super(AbstractPooling, self).__init__()
        self.dim = dim

    def forward(self, x, lens=None):
        raise NotImplementedError()


class MaxPooling(AbstractPooling):
    name = "max"

    def forward(self, x, lens=None):
        if lens is not None:
            mask = utils.mask(lens, x.size(1))
            x = x.masked_fill(1 - mask.unsqueeze(-1), float("-inf"))
        return x.max(1)[0]


class SumPooling(AbstractPooling):
    name = "sum"

    def forward(self, x, lens=None):
        if lens is not None:
            mask = utils.mask(lens, x.size(1))
            x = x.masked_fill(1 - mask.unsqueeze(-1), 0)
        return x.sum(1)


class ProdPooling(AbstractPooling):
    name = "prod"

    def forward(self, x, lens=None):
        if lens is not None:
            mask = utils.mask(lens, x.size(1))
            x = x.masked_fill(1 - mask.unsqueeze(-1), 1)
        return x.prod(1)


class MeanPooling(AbstractPooling):
    name = "mean"

    def forward(self, x, lens=None):
        if lens is None:
            return x.mean(1)
        mask = utils.mask(lens, x.size(1))
        x = x.masked_fill(1 - mask.unsqueeze(-1), 0)
        return x.sum(1) / lens.unsqueeze(-1).float()


class LastPooling(AbstractPooling):
    name = "last"

    def forward(self, x, lens=None):
        if lens is None:
            return x[:, -1]
        idx = lens.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), 1, x.size(-1))
        return x.gather(1, idx - 1).squeeze(1)
