import x2paddle
from x2paddle import torch2paddle
from base import BaseModel
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from itertools import chain
from base import BaseModel
from utils.helpers import initialize_weights
from utils.helpers import set_trainable
from itertools import chain
from models import resnet


def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = (out_channels // 2 if inner_channels is None else
        inner_channels)
    down_conv = nn.Sequential(nn.Conv2D(in_channels, inner_channels,
        kernel_size=3, padding=1, bias_attr=False), nn.BatchNorm2D(
        inner_channels), x2paddle.torch2paddle.ReLU(inplace=True), nn.
        Conv2D(inner_channels, out_channels, kernel_size=3, padding=1,
        bias_attr=False), nn.BatchNorm2D(out_channels), x2paddle.
        torch2paddle.ReLU(inplace=True))
    return down_conv


class encoder(nn.Layer):

    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2D(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x


class decoder(nn.Layer):

    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = torch2paddle.Conv2DTranspose(in_channels, in_channels // 
            2, kernel_size=2, stride=2)
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)
        if x.size(2) != x_copy.size(2) or x.size(3) != x_copy.size(3):
            if interpolate:
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                    mode='bilinear', align_corners=True)
            else:
                diffY = x_copy.size()[2] - x.size()[2]
                diffX = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, 
                    diffY - diffY // 2))
        x = torch2paddle.concat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class UNet(BaseModel):

    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(UNet, self).__init__()
        self.start_conv = x2conv(in_channels, 64)
        self.down1 = encoder(64, 128)
        self.down2 = encoder(128, 256)
        self.down3 = encoder(256, 512)
        self.down4 = encoder(512, 1024)
        self.middle_conv = x2conv(1024, 1024)
        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2D(64, num_classes, kernel_size=1)
        self._initialize_weights()
        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, paddle.nn.Conv2D) or isinstance(module,
                paddle.nn.Linear):
                torch2paddle.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, paddle.nn.BatchNorm2D):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.middle_conv(self.down4(x4))
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.final_conv(x)
        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, paddle.nn.BatchNorm2D):
                module.eval()


"""
-> Unet with a resnet backbone
"""


class UNetResnet(BaseModel):

    def __init__(self, num_classes, in_channels=3, backbone='resnet50',
        pretrained=True, freeze_bn=False, freeze_backbone=False, **_):
        super(UNetResnet, self).__init__()
        model = getattr(resnet, backbone)(pretrained, norm_layer=nn.BatchNorm2d
            )
        self.initial = list(model.children())[:4]
        if in_channels != 3:
            self.initial[0] = nn.Conv2D(in_channels, 64, kernel_size=7,
                stride=2, padding=3, bias_attr=False)
        self.initial = nn.Sequential(*self.initial)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.conv1 = nn.Conv2D(2048, 192, kernel_size=3, stride=1, padding=1)
        self.upconv1 = torch2paddle.Conv2DTranspose(192, 128, 4, 2, 1, bias
            =False)
        self.conv2 = nn.Conv2D(1152, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = torch2paddle.Conv2DTranspose(128, 96, 4, 2, 1, bias=\
            False)
        self.conv3 = nn.Conv2D(608, 96, kernel_size=3, stride=1, padding=1)
        self.upconv3 = torch2paddle.Conv2DTranspose(96, 64, 4, 2, 1, bias=False
            )
        self.conv4 = nn.Conv2D(320, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = torch2paddle.Conv2DTranspose(64, 48, 4, 2, 1, bias=False
            )
        self.conv5 = nn.Conv2D(48, 48, kernel_size=3, stride=1, padding=1)
        self.upconv5 = torch2paddle.Conv2DTranspose(48, 32, 4, 2, 1, bias=False
            )
        self.conv6 = nn.Conv2D(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2D(32, num_classes, kernel_size=1, bias_attr=False)
        initialize_weights(self)
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.
                layer3, self.layer4], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x1 = self.layer1(self.initial(x))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.upconv1(self.conv1(x4))
        x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode='bilinear',
            align_corners=True)
        x = torch2paddle.concat([x, x3], dim=1)
        x = self.upconv2(self.conv2(x))
        x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode='bilinear',
            align_corners=True)
        x = torch2paddle.concat([x, x2], dim=1)
        x = self.upconv3(self.conv3(x))
        x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode='bilinear',
            align_corners=True)
        x = torch2paddle.concat([x, x1], dim=1)
        x = self.upconv4(self.conv4(x))
        x = self.upconv5(self.conv5(x))
        if x.size(2) != H or x.size(3) != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear',
                align_corners=True)
        x = self.conv7(self.conv6(x))
        return x

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(),
            self.layer2.parameters(), self.layer3.parameters(), self.layer4
            .parameters())

    def get_decoder_params(self):
        return chain(self.conv1.parameters(), self.upconv1.parameters(),
            self.conv2.parameters(), self.upconv2.parameters(), self.conv3.
            parameters(), self.upconv3.parameters(), self.conv4.parameters(
            ), self.upconv4.parameters(), self.conv5.parameters(), self.
            upconv5.parameters(), self.conv6.parameters(), self.conv7.
            parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, paddle.nn.BatchNorm2D):
                module.eval()
