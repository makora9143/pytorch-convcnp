import random

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from gpytorch.kernels import RBFKernel, ScaleKernel

from utils import channel_last


class PowerFunction(nn.Module):
    def __init__(self, K=1):
        super().__init__()
        self.K = K

    def forward(self, x):
        return torch.cat(list(map(x.pow, range(self.K + 1))), -1)


class ConvCNP1d(nn.Module):
    def __init__(self, density=16):
        super().__init__()

        self.density = density

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=1)

        self.cnn = nn.Sequential(
            nn.Conv1d(3, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(32, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 2, 5, 1, 2)
        )

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.cnn.apply(weights_init)

        self.pos = nn.Softplus()
        self.psi_rho = ScaleKernel(RBFKernel())

    def forward(self, xc, yc, xt):

        tmp = torch.cat([xc, xt], 1)
        lower, upper = tmp.min(), tmp.max()
        num_t = int((self.density * (upper - lower)).item())
        t = torch.linspace(start=lower, end=upper, steps=num_t).reshape(1, -1, 1).repeat(xc.size(0), 1, 1).to(xc.device)

        h = self.psi(t, xc).matmul(self.phi(yc))
        h0, h1 = h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)
        h = torch.cat([h0, h1], -1)

        rep = torch.cat([t, h], -1).transpose(-1, -2)
        f = self.cnn(rep).transpose(-1, -2)
        f_mu, f_sigma = f.split(1, -1)

        mu = self.psi_rho(xt, t).matmul(f_mu)

        sigma = self.psi_rho(xt, t).matmul(self.pos(f_sigma))
        return MultivariateNormal(mu, scale_tril=sigma.diag_embed())


class ConvCNP2d(nn.Module):
    def __init__(self, channel=1):
        super().__init__()

        self.conv_theta = nn.Conv2d(channel, 128, 9, 1, 4)

        self.cnn = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 1, 1, 0),
            Conv2dResBlock(128, 128),
            Conv2dResBlock(128, 128),
            Conv2dResBlock(128, 128),
            Conv2dResBlock(128, 128),
            nn.Conv2d(128, 2 * channel, 1, 1, 0)
        )

        self.pos = nn.Softplus()

        self.channel = channel

        self.mr = [0.5, 0.7, 0.9]

    def forward(self, I):
        n_total = I.size(2) * I.size(3)
        num_context = int(torch.empty(1).uniform_(n_total / 100, n_total / 2).item())
        M_c = I.new_empty(I.size(0), 1, I.size(2), I.size(3)).bernoulli_(p=num_context / n_total).repeat(1, self.channel, 1, 1)

        signal = I * M_c
        density = M_c

        # self.conv_theta.abs_constraint()
        density_prime = self.conv_theta(density)
        signal_prime = self.conv_theta(signal)
        # signal_prime = signal_prime.div(density_prime + 1e-8)
        # # self.conv_theta.abs_unconstraint()
        h = torch.cat([signal_prime, density_prime], 1)

        f = self.cnn(h)
        mean, std = f.split(self.channel, 1)
        std = self.pos(std)

        mean, std = channel_last(mean), channel_last(std)
        return MultivariateNormal(mean, scale_tril=std.diag_embed())

    def complete(self, I, M_c=None, missing_rate=None):
        if M_c is None:
            if missing_rate is None:
                missing_rate = random.choice(self.mr)
            M_c = I.new_empty(I.size(0), 1, I.size(2), I.size(3)).bernoulli_(p=1 - missing_rate).repeat(1, self.channel, 1, 1)

        signal = I * M_c
        density = M_c

        density_prime = self.conv_theta(density)
        signal_prime = self.conv_theta(signal)
        h = torch.cat([signal_prime, density_prime], 1)

        f = self.cnn(h)
        mean, std = f.split(self.channel, 1)
        std = self.pos(std)
        return M_c, mean, MultivariateNormal(channel_last(mean), scale_tril=channel_last(std).diag_embed())


class Conv2dResBlock(nn.Module):
    def __init__(self, in_channel, out_channel=128):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 1, 2, groups=in_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 5, 1, 2, groups=in_channel),
            nn.ReLU()
        )

        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        output = self.convs(x)
        output = self.final_relu(output + shortcut)
        return output
