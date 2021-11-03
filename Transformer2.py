import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Fcn = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 5, 2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 5, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, 2, bias=False),
            nn.Sigmoid(),
        )
        self.Trn = nn.Transformer(d_model=32, nhead=8, num_decoder_layers=6)

    def forward(self, x):
        out = self.Fcn(x)
        B, C, H, W = out.size()
        out = out.split(C//2, dim=1)
        out1 = out[0]
        out2 = out[1]
        out1 = out1.swapaxes(3, 1).reshape(-1, H, C//2)
        out1 = self.Trn(out1, out1)
        out1 = out1.reshape(-1, W, H, C//2).swapaxes(3, 1)
        out = torch.cat((out1, out2), dim=1)
        return out

if __name__ == "__main__":
    x = torch.randn(2, 3, 128, 128)
    net = Model()
    print(net(x).shape)
