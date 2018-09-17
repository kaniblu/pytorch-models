import logging

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.rnn as R

from .. import common


def init_rnn(cell, gain=1):
    # orthogonal initialization of recurrent weights
    for _, hh, _, _ in cell.all_weights:
        for i in range(0, hh.size(0), cell.hidden_size):
            init.orthogonal_(hh[i:i + cell.hidden_size], gain=gain)


class AbstractRNNCell(common.Module):

    """returns [batch_size, seq_len, hidden_dim]"""

    def __init__(self, input_dim, hidden_dim, *, dynamic=False,
                 layers=1, dropout=0):
        super(AbstractRNNCell, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.dynamic = dynamic
        self.layers = layers
        self.dropout = dropout

    def forward_cell(self, x, h0):
        raise NotImplementedError()

    def form_hidden(self, h0):
        """
        returns hidden state suitable for priming this cell
        :param h0: [batch_size, hidden_dim] Tensor, the first hidden input
        :return:
        """
        batch_size = h0.size(0)
        h = h0.new(self.layers, batch_size, self.hidden_dim).zero_()
        h[0] = h0
        return h

    def forward(self, x, lens=None, h=None):
        """
        :param x: [batch_size x seq_len x input_dim] Tensor
        :param lens: [batch_size] LongTensor
        :param h: [batch_size x hidden_dim] Tensor
        :return:
        """
        batch_size, max_len, _ = x.size()
        if self.dynamic:
            x = R.pack_padded_sequence(x, lens, True)
        o, c, h = self.forward_cell(x, h)
        if self.dynamic:
            o, _ = R.pad_packed_sequence(o, True, 0, max_len)
        return o.contiguous(), c, h


class LSTMCell(AbstractRNNCell):

    name = "lstm-rnn"

    def __init__(self, *args, **kwargs):
        super(LSTMCell, self).__init__(*args, **kwargs)
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
        o, c = self.lstm(x, h0)
        h = c[0].permute(1, 0, 2).contiguous()
        return o, c, h[:, -1]

    def form_hidden(self, h0):
        h = super(LSTMCell, self).form_hidden(h0)
        return (h, torch.zeros_like(h))

    def reset_parameters(self, gain=1):
        self.lstm.reset_parameters()
        init_rnn(self.lstm, gain)


class BidirectionalLSTMCell(LSTMCell):

    name = "bilstm-rnn"

    def _lstm_kwargs(self):
        if self.hidden_dim % 2 != 0:
            logging.warning(f"hidden dim must be multiple of "
                            f"2: {self.hidden_dim}")
        return dict(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=self.layers,
            bidirectional=True,
            dropout=self.dropout,
            batch_first=True
        )

    def form_hidden(self, h0):
        h = super(BidirectionalLSTMCell, self).form_hidden(h0)
        h = h.permute(1, 0, 2).contiguous().view(3, 4, 2).permute(1, 0, 2)
        return (h, torch.zeros_like(h))

    def forward_cell(self, x, h0):
        o, c = self.lstm(x, h0)
        h = c[0].permute(1, 0, 2).contiguous()
        h = h.view(-1, self.layers, 2, self.hidden_dim // 2)
        h = h[:, -1].view(-1, self.hidden_dim)
        return o, c, h


class GRUCell(AbstractRNNCell):

    name = "gru-rnn"

    def __init__(self, *args, **kwargs):
        super(GRUCell, self).__init__(*args, **kwargs)
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
        return o, h, h[:, -1]

    def reset_parameters(self, gain=1):
        self.gru.reset_parameters()
        init_rnn(self.gru, gain)


class BidirectionalGRUCell(GRUCell):

    name = "bigru-rnn"

    def _gru_kwargs(self):
        if self.hidden_dim % 2 != 0:
            logging.warning(f"hidden dim must be multiple of "
                            f"2: {self.hidden_dim}")
        return dict(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            bidirectional=True,
            dropout=self.dropout,
            batch_first=True
        )

    def form_hidden(self, h0):
        h = super(BidirectionalGRUCell, self).form_hidden(h0)
        return h.permute(1, 0, 2).contiguous().view(3, 4, 2).permute(1, 0, 2)

    def forward_cell(self, x, h0):
        o, c = self.gru(x, h0)
        h = c.permute(1, 0, 2).contiguous()
        h = h.view(-1, self.layers, 2, self.hidden_dim // 2)
        h = h[:, -1].view(-1, self.hidden_dim)
        return o, c, h