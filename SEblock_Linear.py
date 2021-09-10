import torch
from torch import nn

#  Linear
class SEblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, input):
        batch_size, c, h, w, = input.size()
        input = self.avg_pool(input)
        input = input.view(batch_size, c)
        input = self.fc(input)
        output = input.view(batch_size, c, 1, 1)
        return output * input





if __name__ == '__main__':
    # pool = nn.AdaptiveAvgPool2d(1)
    # x = torch.randn(1, 3, 4, 4)
    # print(pool(x))
    seblock = SEblock(128)
    x = torch.randn(2, 128, 5, 5)
    print(seblock(x).shape)