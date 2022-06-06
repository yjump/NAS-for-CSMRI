import torch
import torch.nn as nn


def init_weights(modules):
    pass


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.LeakyReLU(inplace=True)
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class D3Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1):
        super(D3Block, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, ksize, stride, padding=2, dilation=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, ksize, stride, padding=3, dilation=3),
            nn.LeakyReLU(inplace=True)
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)+x
        return out


class RDBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RDBlock, self).__init__()

        self.entry = BasicBlock(in_channels, 32, 3, 1, 1)
        self.b1 = D3Block(32, 32, 3, 1)
        self.b2 = D3Block(32, 32, 3, 1)
        self.b3 = D3Block(32, 32, 3, 1)
        self.exit = BasicBlock(32, out_channels, 3, 1, 1)

    def forward(self, x):
        x0 = self.entry(x)
        x1 = self.b1(x0)+x0
        x2 = self.b2(x1)+x0
        x3 = self.b3(x2)+x0
        out = self.exit(x3)
        return out
