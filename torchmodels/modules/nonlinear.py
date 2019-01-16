from .. import common
from . import activation


class AbstractNonlinear(common.Module):

    def __init__(self, in_dim, out_dim=None):
        super(AbstractNonlinear, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim, self.out_dim = in_dim, out_dim

    def forward(self, x):
        raise NotImplementedError


class SimpleNonlinear(AbstractNonlinear):
    name = "simple"

    def __init__(self, *args,
                 activation=activation.ReluActivation, **kwargs):
        super(SimpleNonlinear, self).__init__(*args, **kwargs)
        self.act_cls = activation

        self.linear = common.Linear(self.in_dim, self.out_dim)
        self.act = self.act_cls()

    def forward(self, x):
        x = self.linear(x)
        return self.act(x)


class GatedTanhNonlinear(AbstractNonlinear):
    name = "gated-tanh"

    def __init__(self, *args, **kwargs):
        super(GatedTanhNonlinear, self).__init__(*args, **kwargs)
        self.sigmoid = activation.SigmoidActivation()
        self.tanh = activation.TanhActivation()
        self.o_linear = common.Linear(self.in_dim, self.out_dim)
        self.g_linear = common.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        o = self.o_linear(x)
        g = self.g_linear(x)

        return self.tanh(o) * self.sigmoid(g)
