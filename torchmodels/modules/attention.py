"""
Attention Operation compares a group of query vectors to a candidate of
value vectors and produces an attention vector for each query vector. Then the
value vectors are summed up using the attention weights.

The operands of the attention operation are:

[batch_size x num_queries x qry_dim] Tensor x
    [batch_size x num_values x val_dim] Tensor ->
    [batch_size x num_queries x val_dim] Tensor

All modules in this package support variable number of queries and values.
"""

import math

import torch
import torch.nn as nn

from .. import utils
from .. import common


class AbstractAttention(common.Module):

    def __init__(self, qry_dim, val_dim):
        super(AbstractAttention, self).__init__()
        self.qry_dim = qry_dim
        self.val_dim = val_dim

    def forward(self, qrys, vals, num_vals=None):
        raise NotImplementedError()


class ScaledDotProductOperation(common.Module):

    "arXiv:1706.03762"

    def __init__(self):
        super(ScaledDotProductOperation, self).__init__()
        self.softmax = nn.Softmax(2)

    def forward(self, qrys, keys, vals, num_keyvals=None):
        logits = torch.bmm(qrys, keys.permute(0, 2, 1))
        if num_keyvals is not None:
            scale = 1 / num_keyvals.float().sqrt()
            mask = utils.mask(num_keyvals, keys.size(1)).unsqueeze(1)
            logits *= scale.unsqueeze(-1).unsqueeze(-1)
            logits = logits.masked_fill(1 - mask, float("-inf"))
        else:
            logits *= 1 / math.sqrt(keys.size(1))
        atts = self.softmax(logits)
        return torch.bmm(atts, vals)


class MultiplicativeAttention(AbstractAttention):

    name = "multiplicative-attention"

    def __init__(self, *args, hidden_dim=300, **kwargs):
        super(MultiplicativeAttention, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim

        self.linear_qry = common.Linear(self.qry_dim, self.hidden_dim)
        self.linear_key = common.Linear(self.val_dim, self.hidden_dim)
        self.linear_val = common.Linear(self.val_dim, self.val_dim)
        self.sdp = ScaledDotProductOperation()

    def forward(self, qrys, vals, num_vals=None):
        q, k = self.linear_qry(qrys), self.linear_key(vals)
        v = self.linear_val(vals)
        return self.sdp(q, k, v, num_vals)