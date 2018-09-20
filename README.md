# PyTorch Helper Tools for Models #

This thin wrapper allows models (or more specifically their configurations) 
to be loaded from yaml files. It also promotes easier modularization by separating
high-level hyperparameters (word dimensions) and module-specific hyperparameters (number of hidden layers etc.). Examples can be found in to `tests` folder.

### Features

Load model configuration (not to be confused with parameter loading, i.e. `
torch.load`) from yaml files:

```python
import torch
import torchmodels
from torchmodels.modules import attention

# Contents of att.yml:
# type: multiplicative-attention
# vargs:
#   hidden_dim: 200
model_cls = torchmodels.create_model_cls(attention, model_path="att.yml")
model = model_cls(
    qry_dim=200,
    val_dim=300
)
# [32 x 1 x 300] Tensor
print(model(torch.randn(32, 1, 200), torch.randn(32, 1, 300)).size())
```

Create custom modules and load them using yaml files as well:

```python
import torch.nn as nn
import torchmodels

# assuming that the following code is in a package structure like this:
# models
#  ∟ __init__.py
#  ∟ mlp.py

class MLP(torchmodels.Module):

    name = "some-mlp"

    # use keyword-only arguments
    def __init__(self, input_dim, output_dim, *, 
                 hidden_dim=300, 
                 num_layers=2):
        super(MLP, self).__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tanh = nn.Tanh()

        self.input_layer = torchmodels.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layers = torchmodels.Sequential(
            *[self.nonlinear_cls(self.hidden_dim, self.hidden_dim)
              for _ in range(self.num_layers)]
        )
        self.output_layer = torchmodels.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        return self.output_layer(self.hidden_layers(self.input_layer(x)))

# register packages first
import models
import models.mlp
torchmodels.register_packages(models)

# Contents of mlp.yml:
# type: some-mlp
# vargs:
#   hidden_dim: 200
#   num_layers; 5

# `create_model_cls` will first search a module of the name `some-mlp` under the 
# specified package, then automatically scan keyword-only arguments that are 
# available for fill-in.
# If `model_path` is supplied, arguments in the yaml file will be mapped to
# keyword-only arguments. The return value is a model-initializing function that 
# can be called with positional arguments.
model_cls = torchmodels.create_model_cls(models.mlp, model_path="mlp.yml")
model = model_cls(5, 100)
```

### Scaffolding

Create barebone model configuration file by calling the `scaffold` script:

    # for the previous exmaple
    scaffold models --module-name some-mlp --save-path mlp.yml
    
Barebone configurations can be used as the starting point for building models.


### Install

Install this package through pip: `pip install pytorch-models`


### Disclaimer

There could be many limitations to this approach, but many of my own
deep learning projects have personally benefited from it. Hence I have decided 
to make it public in the form of a library, for someone might find it useful 
as well.
