import torch
import torch.nn as nn
import torch.nn.init as init

from .. import utils
from .. import common


class AbstractEmbedding(common.Module):

    def __init__(self, vocab_size, dim):
        super(AbstractEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim

    def forward(self, x):
        raise NotImplementedError()

    @property
    def weight(self):
        raise NotImplementedError()

    def load(self, idx, tensor):
        raise NotImplementedError()


class TorchEmbedding(nn.Embedding):

    def reset_parameters(self):
        init.xavier_normal_(self.weight.detach())
        if self.padding_idx is not None:
            self.weight.detach()[self.padding_idx].zero_()


class SimpleEmbedding(AbstractEmbedding):
    name = "simple"

    def __init__(self, *args, allow_padding=False, **kwargs):
        super(SimpleEmbedding, self).__init__(*args, **kwargs)
        self.allow_padding = allow_padding
        if self.allow_padding:
            self.pad_idx = self.vocab_size
        else:
            self.pad_idx = None
        self.emb = TorchEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.dim,
            padding_idx=self.pad_idx
        )

    @property
    def num_embeddings(self):
        if self.allow_padding:
            return self.vocab_size + 1
        else:
            return self.vocab_size

    def forward(self, x):
        return self.emb(x)

    @property
    def weight(self):
        return self.emb.weight

    def load(self, idx, tensor):
        assert 0 <= idx < self.vocab_size
        self.emb.weight.detach()[idx] = tensor


def index_map(x, idx):
    size = x.size()
    mapped = torch.index_select(idx, 0, x.view(-1))
    return mapped.view(*size)


class FineTunableEmbedding(AbstractEmbedding):
    name = "fine-tunable"

    def __init__(self, *args, allow_padding=False, freeze=False,
                 frozen_idx=frozenset(), unfrozen_idx=frozenset(), **kwargs):
        super(FineTunableEmbedding, self).__init__(*args, **kwargs)
        self.pad_idx = self.vocab_size
        self.allow_padding = allow_padding
        self.freeze = freeze
        self.frozen_idx = frozen_idx
        self.unfrozen_idx = unfrozen_idx

        if self.frozen_idx is None:
            self.frozen_idx = set()
        if self.unfrozen_idx is None:
            self.unfrozen_idx = set()
        if not isinstance(self.frozen_idx, set):
            self.frozen_idx = set(self.frozen_idx)
        if not isinstance(self.unfrozen_idx, set):
            self.unfrozen_idx = set(self.unfrozen_idx)

        all_idx = set(range(self.vocab_size))
        for i in self.frozen_idx:
            utils.assert_oneof(i, all_idx, "frozen embedding index")
        for i in self.unfrozen_idx:
            utils.assert_oneof(i, all_idx, "unfrozen embedding index")

        if self.freeze:
            self.frozen_idx = all_idx - self.unfrozen_idx
        else:
            self.unfrozen_idx = all_idx - self.frozen_idx
        self.frozen_emb = SimpleEmbedding(
            vocab_size=len(self.frozen_idx),
            dim=self.dim,
            allow_padding=True
        )
        self.unfrozen_emb = SimpleEmbedding(
            vocab_size=len(self.unfrozen_idx),
            dim=self.dim,
            allow_padding=True
        )
        self.frozen_map = common.Parameter(
            data=self._create_frozen_map(),
            requires_grad=False
        )
        self.idx_map = common.Parameter(
            data=self._create_idx_map(),
            requires_grad=False
        )
        self.frozen_emb.emb.weight.requires_grad = False
        self.unfrozen_emb.emb.weight.requires_grad = True

    @property
    def num_embeddings(self):
        if self.allow_padding:
            return self.vocab_size + 1
        else:
            return self.vocab_size

    def _create_frozen_map(self):
        fmap = torch.zeros((self.num_embeddings,)).long()
        fmap[list(self.frozen_idx)] = 1
        if self.allow_padding:
            fmap[self.pad_idx] = 1
        return fmap

    def _create_idx_map(self):
        frozen_idx = list(enumerate(sorted(list(self.frozen_idx))))
        unfrozen_idx = list(enumerate(sorted(list(self.unfrozen_idx))))
        idx = frozen_idx + unfrozen_idx
        idx += [(self.frozen_emb.pad_idx, self.pad_idx)]
        idx.sort(key=lambda x: x[1])
        idx = [x[0] for x in idx]
        return torch.LongTensor(idx)

    def forward(self, x):
        """x: [...] LongTensor"""
        frozen = index_map(x, self.frozen_map)
        unfrozen = 1 - frozen
        idx = index_map(x, self.idx_map)
        x_frozen = frozen * idx + unfrozen * self.frozen_emb.pad_idx
        x_unfrozen = unfrozen * idx + frozen * self.unfrozen_emb.pad_idx
        x_frozen = self.frozen_emb(x_frozen)
        x_unfrozen = self.unfrozen_emb(x_unfrozen)
        return x_frozen + x_unfrozen

    @property
    def weight(self):
        idx = torch.arange(
            self.num_embeddings,
            requires_grad=False,
            device=self.frozen_map.device
        )
        idx = idx.long()
        return self.forward(idx)

    def load(self, idx, tensor):
        assert 0 <= idx < self.vocab_size
        frozen = self.frozen_map[idx]
        idx = self.idx_map[idx]
        if frozen:
            emb = self.frozen_emb
        else:
            emb = self.unfrozen_emb
        emb.weight.detach()[idx] = tensor
