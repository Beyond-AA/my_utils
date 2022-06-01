import torch
from torch import nn
import torch.nn.functional as F


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        output = self.block(x)
        return F.relu(output + x)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn2(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            RestNetBasicBlock(64, 64, 1),
            RestNetBasicBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            RestNetDownBlock(64, 128, [2, 1]),
            RestNetBasicBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            RestNetDownBlock(128, 256, [2, 1]),
            RestNetBasicBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            RestNetDownBlock(256, 512, [2, 1]),
            RestNetBasicBlock(512, 512, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        #  1 * 3 * 1024 * 1024 -> 1 * 61 * 512 * 512
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)
        #  1 * 61 * 512 * 512 ->  1 * 64 * 256 * 256
        out = self.layer1(out)
        #  1 * 64 * 256 * 256 ->  1 * 128 * 128 * 128
        out = self.layer2(out)
        #  1 * 512 * 1
        out = self.layer3(out)
        #  1 * 512
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    x = torch.randn(1, 3, 1024, 1024)
    block = RestNet18()
    print(block(x).shape)