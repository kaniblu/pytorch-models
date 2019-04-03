import torchmodels
from torchmodels.modules import nonlinear


class AbstractMLP(torchmodels.Module):

    def __init__(self, input_dim, output_dim):
        super(AbstractMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError()


class MLP(AbstractMLP):
    name = "mlp"

    def __init__(self, *args, hidden_dims=(200, 100), num_layers=2,
                 nonlinear=nonlinear.SimpleNonlinear, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.nonlinear_cls = nonlinear

        assert len(self.hidden_dims) == self.num_layers

        linear = torchmodels.Linear
        self.input_layer = linear(self.input_dim, self.hidden_dims[0])
        self.hidden_layers = torchmodels.Sequential(
            *[self.nonlinear_cls(a, b)
              for a, b in zip(self.hidden_dims, self.hidden_dims[1:])]
        )
        self.output_layer = linear(self.hidden_dims[-1], self.output_dim)

    def forward(self, x):
        return self.output_layer(self.hidden_layers(self.input_layer(x)))
