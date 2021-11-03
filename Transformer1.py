import torch
from torch import nn
# from torchstat import stat


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.FCN = nn.Sequential(
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
        self.Trn = nn.Transformer(d_model=64, nhead=16, num_decoder_layers=6)

    def forward(self, x):
        out = self.FCN(x)
        B, C, H, W = out.size()
        out = out.swapaxes(3, 1).reshape(-1, H, C)
        out = self.Trn(out, out)
        out = out.reshape(-1, W, H, C).swapaxes(3, 1)
        return out

if __name__ == "__main__":
    x = torch.randn(2, 3, 128, 128)
    net = Model()
    print(net(x).shape)

    # transformer_model = nn.Transformer(nhead=8, num_encoder_layers=6)
    # src = torch.rand((10, 32, 512))
    # tgt = torch.rand((20, 32, 512))
    # out = transformer_model(src, tgt)
    # print(out)
    # stat(transformer_model, src, tgt)
    # print(y.shape)
    # print(sum(param.numel() for param in transformer_model.parameters()))
