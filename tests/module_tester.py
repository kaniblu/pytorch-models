import tqdm
import torch
import torch.nn.functional as F


class ModuleTester(object):

    def __init__(self, mod_cls, input_dim=100, optimizer=torch.optim.Adam,
                 max_iter=1000, show_progress=False, batch_size=32,
                 pass_threshold=0.1):
        self.mod_cls = mod_cls
        self.input_dim = input_dim
        self.optimizer_cls = optimizer
        self.max_iter = max_iter
        self.show_progress = show_progress
        self.batch_size = batch_size
        self.pass_threshold = pass_threshold

    @staticmethod
    def rand_lens(max_len, size, min_len=1):
        return torch.randint(min_len, max_len + 1, size).long()

    @staticmethod
    def check_tensor(x, tensor_type=torch.Tensor, size=None):
        if isinstance(x, (list, tuple)):
            return all(ModuleTester.check_tensor(_x) for _x in x)

        assert isinstance(x, tensor_type), "not a tensor"
        assert (x != x).sum().item() == 0, "NaN detected"
        if size is not None:
            assert x.size() == size

    def test_forward(self):
        model = self.mod_cls(self.input_dim, self.input_dim)
        model.reset_parameters()
        with tqdm.tqdm(total=self.max_iter,
                       disable=not self.show_progress) as t:
            for _ in range(self.max_iter):
                t.update()
                model.reset_parameters()
                input = torch.randn(self.batch_size, self.input_dim)
                output = model(input)
                self.check_tensor(output, size=(self.batch_size, self.input_dim))

    def test_backward(self):
        r"""Tests the model whether it can be trained to learn a simple but noisy
        dataset. The provided model class must be initializable with the input and
        output dimensions.

        Args:
            model_cls (Module): A module class that can be initialized with input
                and output dimensions.
        """

        model = self.mod_cls(self.input_dim, 2)
        model.reset_parameters()
        op = self.optimizer_cls(model.parameters())
        loss = 1
        with tqdm.tqdm(total=self.max_iter,
                       disable=not self.show_progress) as t:
            for _ in range(self.max_iter):
                t.update()
                op.zero_grad()
                data = torch.randint(0, 2, (self.batch_size, self.input_dim))
                data = data.float()
                loss = F.cross_entropy(model(data), data[:, 0].long())
                self.check_tensor(loss)
                loss.backward()
                loss = loss.item()
                t.set_description(f"loss={loss:.3f}")
                if loss < self.pass_threshold:
                    break
                op.step()

        assert loss < self.pass_threshold, \
            f"training through backpropagation failed (loss @ {loss:.3f})"

    def __repr__(self):
        repr = f"<ModuleTester for {inst}>"
        return repr