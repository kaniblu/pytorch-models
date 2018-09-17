import os
import random

import torch
import torchmodels
import torchmodels.manager
import torchmodels.modules.rnn
import torchmodels.modules.pooling

from . import custom_modules
from .custom_modules import nonlinear


BATCH_SIZE = 32
HIDDEN_DIM = 100
YAML_PATH = os.path.join(os.path.dirname(__file__), "rnn.yml")


def test_create_torchmodel_module():
    model_creator = torchmodels.create_model_cls(torchmodels.modules.pooling)
    model = model_creator(HIDDEN_DIM)
    model(torch.randn(BATCH_SIZE, random.randint(3, 10), HIDDEN_DIM))


def test_create_default_nonlinear():
    torchmodels.register_packages(custom_modules)
    model_creator = torchmodels.create_model_cls(custom_modules.nonlinear)
    model = model_creator(HIDDEN_DIM, HIDDEN_DIM)
    model(torch.randn(BATCH_SIZE, HIDDEN_DIM))


def test_create_nonlinear_from_yaml():
    torchmodels.register_packages(custom_modules)
    model_creator = torchmodels.create_model_cls(torchmodels.modules.rnn, YAML_PATH)
    model = model_creator(HIDDEN_DIM, HIDDEN_DIM)
    model(torch.randn(BATCH_SIZE, random.randint(3, 10), HIDDEN_DIM))