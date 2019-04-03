import os

import torch
import torch.nn as nn
import torchmodels
import torchmodels.manager
from torchmodels.modules import rnn
from torchmodels.modules import pooling
from torchmodels.modules import nonlinear
from torchmodels.modules import embedding
from torchmodels.modules import activation
from torchmodels.modules import attention
from torchmodels.modules import feedforward
from torchmodels.modules import relational

from torchmodels import utils
from . import custom_modules
from .custom_modules import mlp
from .module_tester import ModuleTester

YAML_PATHS = [
    os.path.join(os.path.dirname(__file__), "rnn.yml"),
    os.path.join(os.path.dirname(__file__), "mlp.yml")
]
#
#
# def _test_package(pkg, test_fn):
#     for name in torchmodels.manager.get_module_names(pkg):
#         test_fn(name, torchmodels.create_model_cls(pkg, name=name))
#
#
# def _test_attention(name, cls):
#     class Model(torchmodels.Module):
#         num_qrys = 16
#         max_keys = 16
#
#         def __init__(self, in_dim, out_dim, hidden_dim=100):
#             super(Model, self).__init__()
#             self.in_dim, self.out_dim = in_dim, out_dim
#             self.hidden_dim = hidden_dim
#
#             self.qry_linear = nn.Linear(in_dim, hidden_dim * self.num_qrys)
#             self.key_linear = nn.Linear(in_dim, hidden_dim * self.max_keys)
#             self.att = cls(hidden_dim, hidden_dim)
#             self.out_linear = nn.Linear(hidden_dim * self.num_qrys, out_dim)
#
#         def forward(self, x):
#             batch_size = x.size(0)
#             qrys = self.qry_linear(x).view(batch_size, self.num_qrys, -1)
#             keys = self.key_linear(x).view(batch_size, self.max_keys, -1)
#             num_keys = torch.randint(1, self.max_keys + 1, (batch_size,))
#             num_keys = num_keys.long()
#             mask = utils.mask(num_keys, self.max_keys).byte()
#             keys = keys.masked_fill(1 - mask.unsqueeze(-1), float("nan"))
#
#             att = self.att(qrys, keys, num_keys)
#             return self.out_linear(att.view(batch_size, -1))
#
#         def __str__(self):
#             return repr(self)
#
#         def __repr__(self):
#             return f"<Module Encapsulating '{name}'>"
#
#     tester = ModuleTester(Model, max_iter=300)
#     tester.test_backward()
#     tester.test_forward()
#
#
# def test_attention():
#     _test_package(attention, _test_attention)
#
#
# def _test_activation(name, cls):
#     class Model(torchmodels.Module):
#
#         def __init__(self, in_dim, out_dim, hidden_dim=100):
#             super(Model, self).__init__()
#             self.in_dim, self.out_dim = in_dim, out_dim
#             self.hidden_dim = hidden_dim
#
#             self.sequential = torchmodels.Sequential(
#                 nn.Linear(self.in_dim, hidden_dim),
#                 cls(),
#                 nn.Linear(self.hidden_dim, self.out_dim)
#             )
#
#         def forward(self, x):
#             return self.sequential(x)
#
#         def __repr__(self):
#             return f"<Module Encapsulating '{name}'>"
#
#     tester = ModuleTester(Model, max_iter=300, pass_threshold=0.5)
#     tester.test_backward()
#     tester.test_forward()
#
#
# def test_activation():
#     _test_package(activation, _test_activation)
#
#
# def _test_rnn(name, cls):
#     class Model(torchmodels.Module):
#         max_len = 8
#
#         def __init__(self, in_dim, out_dim, hidden_dim=100):
#             super(Model, self).__init__()
#             self.in_dim, self.out_dim = in_dim, out_dim
#             self.hidden_dim = hidden_dim
#
#             self.in_linear = nn.Linear(self.in_dim, self.max_len * hidden_dim)
#             self.rnn = cls(self.in_dim, self.hidden_dim)
#             self.out_linear = nn.Linear(
#                 in_features=(self.max_len + 1) * self.hidden_dim,
#                 out_features=self.out_dim
#             )
#
#         def forward(self, x):
#             batch_size, max_len = x.size(0), self.max_len
#             x = self.in_linear(x).view(batch_size, max_len, -1)
#             lens = torch.randint(1, max_len + 1, (batch_size,)).long()
#             lens = lens.to(x)
#             mask = utils.mask(lens, max_len)
#             x = x.masked_fill(1 - mask.unsqueeze(-1), float("nan"))
#             o, _, h = self.rnn(x, lens)
#             x = torch.cat([o.contiguous().view(batch_size, -1), h], 1)
#             return self.out_linear(x)
#
#         def __repr__(self):
#             return f"<Module Encapsulating '{name}'>"
#
#     tester = ModuleTester(Model, max_iter=300, pass_threshold=0.5)
#     tester.test_backward()
#     tester.test_forward()
#
#
# def test_rnn():
#     _test_package(rnn, _test_rnn)
#
#
# def _test_pooling(name, cls):
#     class Model(torchmodels.Module):
#         max_len = 8
#
#         def __init__(self, in_dim, out_dim):
#             super(Model, self).__init__()
#             self.in_dim, self.out_dim = in_dim, out_dim
#
#             self.in_linear = nn.Linear(self.in_dim, self.max_len * self.in_dim)
#             self.pool = cls(self.in_dim)
#             self.out_linear = nn.Linear(self.in_dim, self.out_dim)
#
#         def forward(self, x):
#             batch_size = x.size(0)
#             lens = torch.randint(1, self.max_len + 1, (batch_size,)).long()
#             mask = utils.mask(lens, self.max_len)
#             x = self.in_linear(x).view(batch_size, self.max_len, -1)
#             x = x.masked_fill(1 - mask.unsqueeze(-1), float("nan"))
#             return self.out_linear(self.pool(x, lens))
#
#         def __repr__(self):
#             return f"<Module Encapsulating '{name}'>"
#
#     tester = ModuleTester(Model, max_iter=500, pass_threshold=0.5)
#     tester.test_backward()
#     tester.test_forward()
#
#
# def test_pooling():
#     _test_package(pooling, _test_pooling)
#
#
# def _test_nonlinear(name, cls):
#     class Model(torchmodels.Module):
#
#         def __init__(self, in_dim, out_dim):
#             super(Model, self).__init__()
#             self.in_dim, self.out_dim = in_dim, out_dim
#
#             self.nonlinear = cls(self.in_dim, self.in_dim)
#             self.out_linear = nn.Linear(self.in_dim, self.out_dim)
#
#         def forward(self, x):
#             return self.out_linear(self.nonlinear(x))
#
#         def __repr__(self):
#             return f"<Module Encapsulating '{name}'>"
#
#     tester = ModuleTester(Model, max_iter=500, pass_threshold=0.5)
#     tester.test_backward()
#     tester.test_forward()
#
#
# def test_nonlinear():
#     _test_package(nonlinear, _test_nonlinear)
#
#
# def _test_feedforward(name, cls):
#     class Model(torchmodels.Module):
#
#         def __init__(self, in_dim, out_dim):
#             super(Model, self).__init__()
#             self.in_dim, self.out_dim = in_dim, out_dim
#
#             self.feedforward = cls(self.in_dim, self.in_dim)
#             self.out_linear = nn.Linear(self.in_dim, self.out_dim)
#
#         def forward(self, x):
#             return self.out_linear(self.feedforward(x))
#
#         def __repr__(self):
#             return f"<Module Encapsulating '{name}'>"
#
#     tester = ModuleTester(Model, max_iter=500, pass_threshold=0.5)
#     tester.test_backward()
#     tester.test_forward()
#
#
# def test_feedforward():
#     _test_package(feedforward, _test_feedforward)
#
#
# def _test_relational(name, cls):
#     class Model(torchmodels.Module):
#         num_keys = 4
#
#         def __init__(self, in_dim, out_dim, hidden_dim=50):
#             super(Model, self).__init__()
#             self.in_dim, self.out_dim = in_dim, out_dim
#             self.hidden_dim = hidden_dim
#
#             self.q_linear = nn.Linear(self.in_dim, self.hidden_dim)
#             self.k_linear = nn.Linear(self.in_dim, self.num_keys * hidden_dim)
#             self.rn = cls(self.hidden_dim, self.hidden_dim, self.hidden_dim)
#             self.out_linear = nn.Linear(self.hidden_dim, self.out_dim)
#
#         def forward(self, x):
#             batch_size = x.size(0)
#             qrys = self.q_linear(x)
#             keys = self.k_linear(x).view(batch_size, self.num_keys, -1)
#             lens = torch.randint(1, self.num_keys + 1, (batch_size,)).long()
#             return self.out_linear(self.rn(qrys, keys, lens))
#
#         def __repr__(self):
#             return f"<Module Encapsulating '{name}'>"
#
#     tester = ModuleTester(Model, max_iter=500, pass_threshold=0.5)
#     tester.test_backward()
#     tester.test_forward()
#
#
# def test_relational():
#     _test_package(relational, _test_relational)
#
#
# def _test_embedding(name, cls):
#     class Model(torchmodels.Module):
#
#         num_words = 10
#
#         def __init__(self, in_dim, out_dim, hidden_dim=100):
#             super(Model, self).__init__()
#             self.in_dim, self.out_dim = in_dim, out_dim
#             self.hidden_dim = hidden_dim
#
#             self.emb = cls(self.num_words, self.hidden_dim)
#             self.pool = torchmodels.Linear(
#                 in_features=self.hidden_dim * 3,
#                 out_features=self.hidden_dim
#             )
#             self.in_linear = torchmodels.Linear(self.in_dim, self.hidden_dim)
#             self.out_linear = torchmodels.Linear(self.hidden_dim, self.out_dim)
#
#         def forward(self, x):
#             batch_size = x.size(0)
#             idx = torch.randint(0, self.num_words, (batch_size, 3))
#             g = torch.sigmoid(self.pool(self.emb(idx).view(batch_size, -1)))
#             return self.out_linear(self.in_linear(x) * g)
#
#         def __repr__(self):
#             return f"<Module Encapsulating '{name}'>"
#
#     tester = ModuleTester(Model, max_iter=500, pass_threshold=0.5)
#     tester.test_backward()
#     tester.test_forward()
#
#
# def test_embedding():
#     _test_package(embedding, _test_embedding)
#
#
# def test_create_rnn_from_yaml():
#     model_cls = torchmodels.create_model_cls(rnn, YAML_PATHS[0])
#     _test_rnn("yaml-loaded-rnn", model_cls)


def test_create_mlp_from_yaml():
    torchmodels.register_packages(custom_modules)
    model_cls = torchmodels.create_model_cls(mlp, YAML_PATHS[1])
    tester = ModuleTester(model_cls, max_iter=300, pass_threshold=0.5)
    tester.test_backward()
    tester.test_forward()
