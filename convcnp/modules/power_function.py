import torch
import torch.nn as nn


class PowerFunction(nn.Module):
    def __init__(self, K=1):
        super().__init__()
        self.K = K

    def forward(self, x):
        return torch.cat(list(map(x.pow, range(self.K + 1))), -1)
