import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.rnn as tr

from .. import common
from .. import utils


def init_rnn(cell, gain=1):
    # orthogonal initialization of recurrent weights
    for _, hh, _, _ in cell.all_weights:
        for i in range(0, hh.size(0), cell.hidden_size):
            init.orthogonal_(hh[i:i + cell.hidden_size], gain=gain)


class AbstractRNN(common.Module):
    r"""An abstract class for RNN modules. Must take an input vector sequence
    and (optionally) its lengths and return hidden outputs at all timesteps,
    and the final hidden state. All derived implementations must abstract away
    the details of RNNs such as producing the final hidden state by
    concatenating a set of bidirectional hidden states from the two ends.

    Shape:
        - Input:
          - sequences: :math: `(N, K, H_{in})` FloatTensor
          - lengths (optional): :math: `(N)` LongTensor
          - initial hidden state (optional): :math: `(N, H_{hid})` FloatTensor
        - Output:
          - hidden states: :math: `(N, K, H_{hid})` FloatTensor
          - final cell state: :math: `(N, H_{hid])` FloatTensor
          - final hidden state: :math: `(N, H_{hid})` FloatTensor

    Minimum Args:
        input_dim (int): Input dimensions
        hidden_dim (int): Hidden state dimensions

    Examples::

        >>> x = torch.randn(16, 8, 100)
        >>> # `FooRNN` is a subclass of `AbstractRNN`
        >>> rnn = FooRNN(100, 200)
        >>> o, c, h = rnn(x)
        >>> o.size(), c.size(), h.size()
        torch.Size([16, 4, 200]), torch.Size([16, 200]), torch.Size([16, 200])
        >>> # the underlying implementation must support variable sequence
        >>> # lengths and initial hidden state
        >>> lens = torch.randint(4, 9, (16, ))
        >>> h0 = torch.randn(16, 200)
        >>> o, c, h = rnn(x, lens, h0)

    """

    def __init__(self, input_dim, hidden_dim):
        super(AbstractRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, lens=None, h=None):
        raise NotImplementedError()


class BaseRNN(AbstractRNN):

    def __init__(self, *args, dynamic=False, layers=1, dropout=0, **kwargs):
        super(BaseRNN, self).__init__(*args, **kwargs)
        self.dynamic = dynamic
        self.layers = layers
        self.dropout = dropout

    def forward_cell(self, x, h0):
        raise NotImplementedError()

    def forward(self, x, lens=None, h=None):
        batch_size, max_len, _ = x.size()
        mask = None
        if lens is not None:
            mask = utils.mask(lens, max_len)
            x = x.masked_fill(1 - mask.unsqueeze(-1), 0)
        if self.dynamic:
            x = tr.pack_padded_sequence(x, lens, True)
        o, c, h = self.forward_cell(x, h)
        if self.dynamic:
            o, _ = tr.pad_packed_sequence(o, True, 0, max_len)
            o = o.contiguous()
        return o, c, h


class LSTM(BaseRNN):
    name = "lstm"

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        self.lstm = nn.LSTM(**self._lstm_kwargs())

    def _lstm_kwargs(self):
        return dict(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            bidirectional=False,
            dropout=self.dropout,
            batch_first=True
        )

    def forward_cell(self, x, h0):
        o, (h, c) = self.lstm(x, h0)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        return o, c[:, -1], h[:, -1]

    def reset_parameters(self, gain=1):
        self.lstm.reset_parameters()
        init_rnn(self.lstm, gain)


class BidirectionalLSTM(LSTM):
    name = "bilstm"

    @property
    def cell_hidden_dim(self):
        return self.hidden_dim // 2

    def _lstm_kwargs(self):
        if self.hidden_dim % 2 != 0:
            raise ValueError(f"bidirectional LSTM only accepts even-numbered "
                             f"hidden dimensions: {self.hidden_dim}")
        return dict(
            input_size=self.input_dim,
            hidden_size=self.cell_hidden_dim,
            num_layers=self.layers,
            bidirectional=True,
            dropout=self.dropout,
            batch_first=True
        )

    def forward_cell(self, x, h0):
        o, (h, c) = self.lstm(x, h0)
        h = h.view(self.layers, 2, -1, self.cell_hidden_dim)
        h = h.permute(2, 0, 1, 3).contiguous()
        c = c.view(self.layers, 2, -1, self.cell_hidden_dim)
        c = c.permute(2, 0, 1, 3).contiguous()
        h = h[:, -1].view(-1, self.hidden_dim)
        c = c[:, -1].view(-1, self.hidden_dim)
        return o, c, h


class GRU(BaseRNN):
    name = "gru"

    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__(*args, **kwargs)
        self.gru = nn.GRU(**self._gru_kwargs())

    def _gru_kwargs(self):
        return dict(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            bidirectional=False,
            dropout=self.dropout,
            batch_first=True
        )

    def forward_cell(self, x, h0):
        o, h = self.gru(x, h0)
        h = h.permute(1, 0, 2).contiguous()
        return o, h[:, -1], h[:, -1]

    def reset_parameters(self, gain=1):
        self.gru.reset_parameters()
        init_rnn(self.gru, gain)


class BidirectionalGRU(GRU):
    name = "bigru"

    @property
    def cell_hidden_dim(self):
        return self.hidden_dim // 2

    def _gru_kwargs(self):
        if self.hidden_dim % 2 != 0:
            raise ValueError(f"bidirectional GRU only accepts even-numbered "
                             f"hidden dimensions: {self.hidden_dim}")
        return dict(
            input_size=self.input_dim,
            hidden_size=self.cell_hidden_dim,
            num_layers=self.layers,
            bidirectional=True,
            dropout=self.dropout,
            batch_first=True
        )

    def forward_cell(self, x, h0):
        o, h = self.gru(x, h0)
        h = h.view(self.layers, 2, -1, self.cell_hidden_dim)
        h = h.permute(2, 0, 1, 3).contiguous()
        h = h.view(-1, self.layers, self.hidden_dim)
        return o, h[:, -1], h[:, -1]
