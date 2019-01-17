import torch

from . import feedforward
from . import pooling
from .. import common
from .. import utils


class AbstractRelationalNetwork(common.Module):
    """Relational networks accept a pair of query [batch_size x query_dim]
    and key [batch_size x num_keys x key_dim] tensors and outputs
    result [batch_size x key_dim] tensor.
    """

    def __init__(self, qry_dim, key_dim, out_dim):
        super(AbstractRelationalNetwork, self).__init__()
        self.qry_dim = qry_dim
        self.key_dim = key_dim
        self.out_dim = out_dim

    def forward(self, qry, keys, num_keys=None):
        raise NotImplementedError()


class RelationalNetwork(AbstractRelationalNetwork):
    name = "simple"

    def __init__(self, *args,
                 hidden_dim=300,
                 relational_mlp=feedforward.MultiLayerFeedForward,
                 output_mlp=feedforward.MultiLayerFeedForward,
                 pooling=pooling.MeanPooling, **kwargs):
        super(RelationalNetwork, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.relational_mlp_cls = relational_mlp
        self.output_mlp_cls = output_mlp
        self.pooling_cls = pooling

        self.relational_mlp = self.relational_mlp_cls(
            input_dim=self.qry_dim + self.key_dim * 2,
            output_dim=self.hidden_dim
        )
        self.output_mlp = self.output_mlp_cls(
            input_dim=self.hidden_dim,
            output_dim=self.out_dim
        )
        self.pooling = self.pooling_cls(self.hidden_dim)

    def relate(self, qry, x, lens=None):

        def all_pair(x):
            batch_size, k, dim = x.size()
            x_exp = x.unsqueeze(2).expand(batch_size, k, k, dim)
            h1 = x_exp.contiguous().view(batch_size, -1, dim)
            h2 = x_exp.permute(0, 2, 1, 3).contiguous().view(batch_size, -1,
                                                             dim)
            return torch.cat([h1, h2], 2)

        batch_size, k, dim = x.size()
        ap = all_pair(x)

        if lens is not None:
            mask = utils.mask(lens, k)
            mask_ap = all_pair(mask.unsqueeze(-1))
            mask = mask_ap.prod(-1).squeeze(-1)
            # mask: [batch_size, k * k]
        else:
            mask = None

        return ap, mask

    def forward(self, qry, keys, num_keys=None):
        key_ap, pair_mask = self.relate(qry, keys, num_keys)
        batch_size, k, ap_dim = key_ap.size()
        qry_exp = qry.unsqueeze(1).expand(batch_size, k, self.qry_dim)
        jnt = torch.cat([key_ap, qry_exp], 2)
        jnt_flat = jnt.view(-1, self.qry_dim + ap_dim)
        jnt_flat = self.relational_mlp(jnt_flat)
        rels = jnt_flat.view(batch_size, -1, self.hidden_dim)
        pair_mask, idx = pair_mask.sort(1, True)
        rels = rels.gather(1, idx.unsqueeze(-1).expand_as(rels))
        out = self.pooling(rels, pair_mask.sum(1))
        return self.output_mlp(out)
