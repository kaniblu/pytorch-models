import torch.nn as nn

from .. import common
from . import activation


class AbstractFeedForward(common.Module):

    def __init__(self, input_dim, output_dim):
        super(AbstractFeedForward, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, h):
        raise NotImplementedError()


class MultiLayerFeedForward(AbstractFeedForward):
    name = "multilayer"

    def __init__(self, *args,
                 num_layers=1,
                 hidden_dim=300,
                 activation=activation.ReluActivation,
                 dropout=0.0,
                 batch_norm=False, **kwargs):
        super(MultiLayerFeedForward, self).__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation_cls = activation
        self.dropout_prob = dropout
        self.should_dropout = dropout > 0.0
        self.should_batchnorm = batch_norm

        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.should_batchnorm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(self.activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.sequential = nn.Sequential(*layers)

        self.reset_parameters()

    def forward(self, h):
        return self.sequential(h)
