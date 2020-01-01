import torch
from torch.utils import data as tdata
from gpytorch.utils.cholesky import psd_safe_cholesky

from .kernels import eq_kernel, matern_kernel, periodic_kernel


class Synthetic1D(tdata.Dataset):
    def __init__(self,
                 kernel,
                 length_scale=1.0,
                 output_scale=1.0,
                 num_total_max=50,
                 random_params=False,
                 train=True,
                 data_range=(-2, 2),
                 ):

        self.x_dim = 1
        self.y_dim = 1

        if kernel == 'eq':
            self.kernel = eq_kernel
        elif kernel == 'matern':
            self.kernel = matern_kernel
        elif kernel == 'periodic':
            self.kernel = periodic_kernel
        else:
            raise NotImplementedError('{} kernel is not implemented'.format(kernel))

        self.length_scale = length_scale
        self.output_scale = output_scale

        self.num_total_max = num_total_max

        self.random_params = random_params
        self.train = train

        self.data_range = data_range

        self.length = 256 if self.train else 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        num_context = torch.randint(3, self.num_total_max, size=())
        num_target = torch.randint(3, self.num_total_max, size=())
        return self.sample(num_context, num_target)

    def set_length(self, batch_size):
        self.length *= batch_size

    def sample(self, num_context, num_target):
        r"""
        Args:
            num_context (int): Number of context points at the sample.
            num_target (int): Number of target points at the sample.

        Returns:
            :class:`Tensor`.
                Different between train mode and test mode:

                *`train`: `num_context x x_dim`, `num_context x y_dim`, `num_total x x_dim`, `num_total x y_dim`
                *`test`: `num_context x x_dim`, `num_context x y_dim`, `400 x x_dim`, `400 x y_dim`
        """
        if self.train:
            num_total = num_context + num_target
            x_values = torch.empty(num_total, self.x_dim).uniform_(*self.data_range)
        else:
            lower, upper = self.data_range
            num_total = int((upper - lower) / 0.01 + 1)
            x_values = torch.linspace(self.data_range[0], self.data_range[1], num_total).unsqueeze(-1)

        if self.random_params:
            length_scale = torch.empty(self.y_dim, self.x_dim).uniform_(
                0.1, self.length_scale)  # [y, x]
            output_scale = torch.empty(self.y_dim).uniform_(0.1, self.output_scale)  # [y]
        else:
            length_scale = torch.full((self.y_dim, self.x_dim), self.length_scale)
            output_scale = torch.full((self.y_dim,), self.output_scale)

        # [y_dim, num_total, num_total]
        covariance = self.kernel(x_values, length_scale, output_scale)

        cholesky = psd_safe_cholesky(covariance)

        # [num_total, num_total] x [] = []
        y_values = cholesky.matmul(torch.randn(self.y_dim, num_total, 1)).squeeze(2).transpose(0, 1)

        if self.train:
            context_x = x_values[:num_context, :]
            context_y = y_values[:num_context, :]
        else:
            idx = torch.randperm(num_total)
            context_x = torch.gather(x_values, 0, idx[:num_context].unsqueeze(-1))
            context_y = torch.gather(y_values, 0, idx[:num_context].unsqueeze(-1))

        return context_x, context_y, x_values, y_values


class _CustomMapDatasetFetcher(tdata._utils.fetch._BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            num_context = torch.randint(3, self.dataset.num_total_max, size=())
            num_target = torch.randint(3, self.dataset.num_total_max, size=())
            data = [self.dataset.sample(num_context, num_target) for _ in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


tdata._utils.fetch._MapDatasetFetcher.fetch = _CustomMapDatasetFetcher.fetch
