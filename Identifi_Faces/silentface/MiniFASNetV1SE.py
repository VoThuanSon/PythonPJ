
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.PReLU(inp),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup),
    )

class MiniFASNetV1SE(nn.Module):
    def __init__(self, num_classes=2, input_size=(80, 80), width_mult=1.0):
        super().__init__()
        self.conv1 = conv_bn(3, int(32 * width_mult), 1)
        self.conv2_dw = conv_dw(int(32 * width_mult), int(64 * width_mult), 1)
        self.fc = nn.Linear(int(64 * width_mult), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
