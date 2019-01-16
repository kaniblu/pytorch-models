import torch

from .. import common


class AbstractActivation(common.Module):

    def forward(self, x):
        raise NotImplementedError()


class ReluActivation(AbstractActivation):
    name = "relu"

    def forward(self, x):
        return torch.relu(x)


class TanhActivation(AbstractActivation):
    name = "tanh"

    def forward(self, x):
        return torch.tanh(x)


class SigmoidActivation(AbstractActivation):
    name = "sigmoid"

    def forward(self, x):
        return torch.sigmoid(x)
