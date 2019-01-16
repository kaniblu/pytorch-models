import math

import torch
import torch.nn as nn

from .. import utils
from .. import common


class AbstractAttention(common.Module):
    r"""An abstract class for attention operations. An attention operation
    takes a pair of query vectors and key vectors and produces attended key
    vectors. For each item in the mini-batch, i-th attended output vector must
    be the attention result of i-th query vector over all key vectors in the
    batch item. The underlying implementation could come with different
    flavors, with multiplicative attention being the most popular one
    (arXiv:1706.03762).

    Shape:
        - Input:
          - query vectors: :math: `(N, K_{qry}, H_{qry})` FloatTensor
          - key vectors: :math: `(N, K_{key}, H_{key})` FloatTensor
          - number of keys: :math: `(N)` LongTensor
        - Output:
          - attended vectors: :math: `(N, K_{qry}, H_{key})` FloatTensor

    Minimum Args:
        qry_dim (int): Dimension of the query vectors
        val_dim (int): Dimension of the key/value vectors (they're the same)

    Examples::

        >>> # 4 query vectors for each data sample
        >>> qry = torch.randn(16, 4, 100)
        >>> # 8 key vectors for each data sample
        >>> key = torch.randn(16, 8, 200)
        >>> # `FooAttention` is a subclass of `AbstractAttention`
        >>> att = FooAttention(100, 200)
        >>> res = att(qry, key)
        >>> res.size()
        torch.Size([16, 4, 200])
        >>> # the underlying implementation must support variable key sizes
        >>> # we randomly generate key sizes from [4, 9) to demonstrate
        >>> num_keys = torch.randint(4, 9, (16, ))
        >>> # attention performed only on first specified number of keys in
        >>> # each data sample
        >>> res = att(qry, key, num_keys)
        >>> res.size()
        torch.Size([16, 4, 200])

    """

    def __init__(self, qry_dim, val_dim):
        super(AbstractAttention, self).__init__()
        self.qry_dim = qry_dim
        self.val_dim = val_dim

    def forward(self, qrys, vals, num_vals=None):
        raise NotImplementedError()


class ScaledDotProductOperation(common.Module):
    """arXiv:1706.03762"""

    def __init__(self):
        super(ScaledDotProductOperation, self).__init__()
        self.softmax = nn.Softmax(2)

    def forward(self, qrys, keys, vals, num_keyvals=None):
        logits = torch.bmm(qrys, keys.permute(0, 2, 1))
        if num_keyvals is not None:
            scale = 1 / num_keyvals.float().sqrt()
            mask = utils.mask(num_keyvals, keys.size(1))
            logits *= scale.unsqueeze(-1).unsqueeze(-1)
            logits = logits.masked_fill(1 - mask.unsqueeze(1), float("-inf"))
            atts = self.softmax(logits)
            vals = vals.masked_fill(1 - mask.unsqueeze(-1), 0)
            return torch.bmm(atts, vals)
        else:
            logits *= 1 / math.sqrt(keys.size(1))
            atts = self.softmax(logits)
            return torch.bmm(atts, vals)


class MultiplicativeAttention(AbstractAttention):
    name = "multiplicative"

    def __init__(self, *args, hidden_dim=300, **kwargs):
        super(MultiplicativeAttention, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim

        self.linear_qry = common.Linear(self.qry_dim, self.hidden_dim)
        self.linear_key = common.Linear(self.val_dim, self.hidden_dim)
        self.linear_val = common.Linear(self.val_dim, self.val_dim)
        self.sdp = ScaledDotProductOperation()

    def forward(self, qrys, vals, num_vals=None):
        if num_vals is not None:
            mask = utils.mask(num_vals, vals.size(1))
            vals = vals.masked_fill(1 - mask.unsqueeze(-1), 0)
        q, k = self.linear_qry(qrys), self.linear_key(vals)
        v = self.linear_val(vals)
        return self.sdp(q, k, v, num_vals)
