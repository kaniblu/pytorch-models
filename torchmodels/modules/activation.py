import torch.nn.functional as tf

from .. import common


class AbstractActivation(common.Module):
    def forward(self, x):
        raise NotImplementedError()


class ReluActivation(AbstractActivation):
    name = "relu-activation"

    def forward(self, x):
        return tf.relu(x)


class TanhActivation(AbstractActivation):
    name = "tanh-activation"

    def forward(self, x):
        return tf.tanh(x)


class SigmoidActivation(AbstractActivation):
    name = "sigmoid-activation"

    def forward(self, x):
        return tf.sigmoid(x)