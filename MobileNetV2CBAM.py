import torch.nn as nn
import math
from CBAM import CBAM

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
#         nn.ELU(inplace=True)
        nn.ReLU6(inplace=True),
#         CoordAtt(inp=oup, oup=oup),
#         SqueezeExciteBlock(oup)
        #Mish(),
        # nn.PReLU()
#         CBAM(oup)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
#         nn.ELU(inplace=True)
        nn.ReLU6(inplace=True)
        #Swish()
        #Mish(),
        # nn.PReLU()

    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_cbam=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_cbam = use_cbam

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if self.use_res_connect is True:
            self.cbam = CBAM(oup)
#             self.cbam = CoordAtt(inp=inp, oup=inp)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
#                 nn.ELU(inplace=True),
                #DYReLU(hidden_dim),
                #Mish(),
                # nn.PReLU(),
                nn.ReLU6(inplace=True),
                # CBAM(hidden_dim),
#                 eca_layer(hidden_dim),
                #Swish(),
                #PReLU(num_parameters=hidden_dim), #nn.LeakyReLU(0.2),
#                 CoordAtt(inp=hidden_dim, oup=hidden_dim),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # CBAMnew(oup),

                nn.BatchNorm2d(oup),
                # BAM(oup)
                #CBAMWithSE(oup),
                # SEBlock(oup)
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
#                 nn.ELU(inplace=True),
                nn.ReLU6(inplace=True),

                # nn.PReLU(),
                #Mish(),
                #DYReLU(hidden_dim),
                #nn.LeakyReLU(0.2),
                # nn.ELU(inplace=True),
                #Swish(),#
                #PReLU(num_parameters=hidden_dim),#CBAM(hidden_dim),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
#                 CBAM(hidden_dim),
                #DYReLU(hidden_dim),
                #Mish(),
#                 nn.ELU(inplace=True),
                nn.ReLU6(inplace=True),
#                 CBAM(hidden_dim),
#                 eca_layer(hidden_dim),

                # nn.PReLU(),
                #Swish(),#
                #PReLU(num_parameters=hidden_dim),#nn.LeakyReLU(0.2), #nn.ELU(inplace=True),
#                 CBAM(hidden_dim),
#                 CoordAtt(inp=hidden_dim, oup=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # CBAMnew(oup),
                nn.BatchNorm2d(oup),
                # BAM(oup),
                # SEBlock(oup)

            )



    def forward(self, x):
        out = self.conv(x)

#         if self.use_cbam is True:
#             out = self.cbam(out)

        if self.use_res_connect:
#             return x + out
            return x + self.cbam(x) + out
        else:
            return out


class MobileNetV2(nn.Module):
    def __init__(self, n_class=8, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        #cbm = cbam
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t -> expansion factor,
            # c -> output channel,
            # n -> number of loop,
            # s -> stride
            [1, 16, 1, 1, False],
            [6, 24, 2, 2, False],
            [6, 32, 3, 2, True],
            [6, 64, 4, 2, False],
            [6, 96, 3, 1, True],
            [6, 160, 3, 2, True],
            [6, 320, 1, 1, True],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
#         self.features.append(CBAM(input_channel))
        # building inverted residual blocks
        for t, c, n, s, cbam in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    #self.features.append(cbam(input_channel))
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, use_cbam=cbam))
                else:
                    #self.features.append(cbam(input_channel))
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, use_cbam=cbam))

                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
#         self.features.append(CBAM(self.last_channel))
        # self.features.append(SqueezeExcitationBlock(self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
#         self.dropout = nn.Dropout(0.4)

        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
#         x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                  m.bias.data.zero_()