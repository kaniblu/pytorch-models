import os
import random

import torch
import torchmodels
import torchmodels.manager
from torchmodels.modules import rnn
from torchmodels.modules import pooling
from torchmodels.modules import embedding
from torchmodels.modules import activation
from torchmodels.modules import gaussian

from . import custom_modules
from .custom_modules import nonlinear
from .custom_modules import mlp


BATCH_SIZE = 32
HIDDEN_DIM = 100
YAML_PATHS = [
    os.path.join(os.path.dirname(__file__), "rnn.yml"),
    os.path.join(os.path.dirname(__file__), "mlp.yml")
]



def test_create_torchmodel_pooling():
    model_creator = torchmodels.create_model_cls(torchmodels.modules.pooling)
    model = model_creator(HIDDEN_DIM)
    model(torch.randn(BATCH_SIZE, random.randint(3, 10), HIDDEN_DIM))


def test_create_torchmodel_activation():
    model_creator = torchmodels.create_model_cls(torchmodels.modules.activation)
    model = model_creator()
    model(torch.randn(BATCH_SIZE, random.randint(3, 10), HIDDEN_DIM))


def test_create_torchmodel_embedding():
    model_creator = torchmodels.create_model_cls(torchmodels.modules.embedding)
    model = model_creator(100, HIDDEN_DIM)
    model(torch.randint(0, 100, (BATCH_SIZE, )).long())


def test_create_torchmodel_gaussian():
    model_creator = torchmodels.create_model_cls(torchmodels.modules.gaussian)
    model = model_creator(HIDDEN_DIM, HIDDEN_DIM)
    model(torch.randn(BATCH_SIZE, HIDDEN_DIM))


def test_create_default_nonlinear():
    torchmodels.register_packages(custom_modules)
    model_creator = torchmodels.create_model_cls(custom_modules.nonlinear)
    model = model_creator(HIDDEN_DIM, HIDDEN_DIM)
    model(torch.randn(BATCH_SIZE, HIDDEN_DIM))


def test_create_rnn_from_yaml():
    torchmodels.register_packages(custom_modules)
    model_creator = torchmodels.create_model_cls(torchmodels.modules.rnn, YAML_PATHS[0])
    model = model_creator(HIDDEN_DIM, HIDDEN_DIM)
    model(torch.randn(BATCH_SIZE, random.randint(3, 10), HIDDEN_DIM))


def test_create_mlp_from_yaml():
    torchmodels.register_packages(custom_modules)
    model_creator = torchmodels.create_model_cls(mlp, YAML_PATHS[1])
    model = model_creator(HIDDEN_DIM, HIDDEN_DIM)
    model(torch.randn(BATCH_SIZE, HIDDEN_DIM))