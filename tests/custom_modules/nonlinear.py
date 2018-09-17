import torchmodels
import torch.nn as nn
import torch.nn.init as init


class AbstractNonlinear(torchmodels.Module):
    def __init__(self, in_dim, out_dim=None):
        super(AbstractNonlinear, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim, self.out_dim = in_dim, out_dim


class FunctionalNonlinear(AbstractNonlinear):
    def __init__(self, *args, **kwargs):
        super(FunctionalNonlinear, self).__init__(*args, **kwargs)
        self.linear = torchmodels.Linear(self.in_dim, self.out_dim)
        self.func = self.get_func()

    @classmethod
    def get_func(cls):
        raise NotImplementedError()

    def forward(self, x):
        x = self.linear(x)
        return self.func(x)


class TanhNonlinear(FunctionalNonlinear):
    name = "tanh"

    def get_func(cls):
        return nn.Tanh()


class ReluNonlinear(FunctionalNonlinear):
    name = "relu"

    def get_func(cls):
        return nn.ReLU()


class GatedTanhNonlinear(AbstractNonlinear):
    name = "gated-tanh"

    def __init__(self, *args, **kwargs):
        super(GatedTanhNonlinear, self).__init__(*args, **kwargs)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.o_linear = nn.Linear(self.in_dim, self.out_dim)
        self.g_linear = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        o = self.o_linear(x)
        g = self.g_linear(x)

        return self.tanh(o) * self.sigmoid(g)

    def reset_parameters(self):
        init.xavier_normal_(self.o_linear.weight.detach())
        init.xavier_normal_(self.g_linear.weight.detach())
        self.o_linear.bias.detach().zero_()
        self.g_linear.bias.detach().zero_()