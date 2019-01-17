# A Management Tool for PyTorch Modules #

![Build Status](https://travis-ci.com/kaniblu/pytorch-models.svg?branch=master)

This library provides a thin wrapper for pytorch modules, allowing them to be 
directly created from yaml/json specification files and thereby improving code
readability/modularization. The library also comes with a few modules 
pre-implemented, so be sure to check it out. 


## Motivation ##

A standard machine learning development pipeline at least consists of following
steps:

  1. model development
  2. parameter optimization through experimentation
  3. application

When using pytorch to accomplish all steps above, there could be two 
common software engineering issues:

  1. the inability to directly create and manipulate models from a configuration
     file
  2. glue code (for model manipulation) repetition for different tasks 
     (e.g. training, testing, predicting, etc.)
     
With this tool, one could reduce code repetition and improve code readability
by just by changing a few lines of code.
 

## Philosophy ##

`pytorch-models` uses strong modularization as the basic design philosophy, 
 meaning that modules will be grouped by their extrinsic properties, i.e.
 input and output shapes. The initialization arguments directly reflects this
 design philosophy: positional arguments are reserved for parameters that 
 influence its extrinsic behaviors and keyword arguments are reserved for 
 parameters that are related to module-specific behaviors. 
 
 As a basic example, a multi-layer feedforward module will always take 
 input vectors of certain dimensions and produce output vectors of certain
 dimensions. Those input and output parameters will be invariant, regardless
 of the underlying implementation. However, the number of layers, whether to use
 batch normalization, etc. can be varied depending on the study results. 
 
 This is a straight-forward design philosophy that we all are aware of, but this library tool will help you to strongly enforce it in all module designs. 
 
 
## Install ##

Install this package using pip.

    pip install pytorch-models


## Features ##

### Loading Model Configurations ###

Suppose that you have a yaml file saved at `att.yml` with the following 
contents.

```yaml
type: multiplicative-attention
vargs:
  hidden_dim: 200
```

Following python script will create the model from the configuration file above.

```python
import torch
import torchmodels
from torchmodels.modules import attention

model_cls = torchmodels.create_model_cls(attention, model_path="att.yml")
model = model_cls(
    qry_dim=200,
    val_dim=300
)
model(torch.randn(32, 1, 200), torch.randn(32, 1, 300)).size()
# [32 x 1 x 300] Tensor

```

### Creating Custom Modules ###

Create custom modules by defining a module as a subclass of `torchmodels.Module`
instead of `torch.Module`. 

In order to make the library aware of the custom module, the root package must 
be registered using `torchmodels.register_packages` function. For example, let's
assume that multilayer feedforward module is defined in `mlp.py` and placed in
a package structure as follows.

```
models
  ∟ __init__.py
  ∟ mlp.py
```

The following script registers the newly created module.

```
import torchmodels
torchmodels.register_packages(models)
```
    
Let's assume that the new module in `mlp.py` has the following initialization 
signature.

```python
import torchmodels

class MLP(torchmodels.Module):

    name = "some-mlp"

    def __init__(self, input_dim, output_dim, *, 
                 hidden_dim=300, num_layers=2):
        super(MLP, self).__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        ...

    def forward(self, x):
        ...
        return ...
```

Then, the configuration file only needs to specify keyword-only arguments:

```yaml
# mlp.yml
type: some-mlp
vargs:
  hidden_dim: 200
  num_layers: 5
```

The configuration file can be used to create the model initialization
**function**, which requires user to supply with positional arguments to obtain
the model, as illustrated below.

```python
# register packages first
import models
import torchmodels

torchmodels.register_packages(models)

model_cls = torchmodels.create_model_cls(models.mlp, model_path="mlp.yml")
model = model_cls(5, 100)
```

Under the hood, `create_model_cls` will first search a module of the name
`some-mlp` defined in the specified package, then fill-in keyword-only arguments
with those under `vargs` key in the specification file.

### Scaffolding ###

Creating configuration files from scratch can be cumbersome. Create barebone model configuration files by calling `scaffold` command:

```
# for the previous model example
scaffold models --module-name some-mlp --save-path mlp.yml
```

### Pre-defined Modules ###

There are some handy modules that can be used on the fly. These
modules can be found under `torchmodels.modules`, and they are 
pre-registered through `torchmodels.register_packages`. Some useful
modules include `attention` modules and `pooling` modules. Both 
modules support variable number of items across a mini-batch. Details
are included in the respective abstract class docstring.
