import torch.nn as nn


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
