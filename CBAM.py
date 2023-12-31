import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1))

        #self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)

        self.relu1 = nn.ReLU(inplace=True)
        #self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)

        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = self.avg_pool(x).squeeze(-1).squeeze(-1)
        max = self.max_pool(x).squeeze(-1).squeeze(-1)

        avg = self.fc2(self.relu1(self.fc1(avg)))
        max = self.fc2(self.relu1(self.fc1(max)))

        out = avg + max

        return self.sigmoid(out).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
#         self.bn1 = nn.BatchNorm2d(1)
#         self.act = h_swish()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max = torch.max(x, dim=1, keepdim=True)[0]
        out = torch.cat([avg, max], dim=1)
        out = self.conv(out)
#         out = self.bn1(out)
        return  self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=3):
        super(CBAM, self).__init__()

        self.channel_gate = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_gate = SpatialAttention(kernel_size)
        #self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        ret = self.channel_gate(x) * x
        retu = self.spatial_gate(ret) * ret

        return retu