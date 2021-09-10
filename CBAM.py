import torch
from torch import nn
class ChannelAtteintion(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAtteintion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        avg_output = self.fc2(self.relu1(self.fc1(self.avg_pool(input))))
        max_output = self.fc2(self.relu1(self.fc1(self.max_pool(input))))
        out = self.sigmoid(avg_output + max_output)
        return out * input

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)  #  "kerneal_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.con1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        avg_out = torch.mean(input, dim=1, keepdim=True)
        max_out, _ = torch.max(input, dim=1, keepdim=True)
        input = torch.cat([avg_out, max_out], dim=1)
        input = self.con1(input)
        output = self.sigmoid(input)
        return output

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAtteintion(channel)
        self.spatialattention = SpatialAttention()
    def forward(self, input):
        x = self.spatialattention(self.channelattention(input))
        out = x * input
        return out



if __name__ == '__main__':
    #cattention = ChannelAtteintion(16)
    cbam = CBAM(128)
    x = torch.randn(2, 128, 16, 16)
    print(cbam(x).shape)

