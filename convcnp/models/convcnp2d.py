import random

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from ..modules import Conv2dResBlock
from ..utils import channel_last


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


