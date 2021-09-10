import torch
from torch import nn

#  Con2d
class SEblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.con = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.avg_pool(input)
        output = self.con(input)
        return output * input

if __name__ == '__main__':
    # pool = nn.AdaptiveAvgPool2d(1)
    # x = torch.randn(1, 3, 4, 4)
    # print(pool(x))
    seblock = SEblock(128)
    x = torch.randn(2, 128, 5, 5)
    print(seblock(x).shape)