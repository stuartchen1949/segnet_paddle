import x2paddle
from x2paddle import torch2paddle
from base import BaseModel
import paddle.nn as nn
import paddle.nn.functional as F
from x2paddle import models
from utils.helpers import get_upsampling_weight
import paddle
from itertools import chain


class FCN8(BaseModel):

    def __init__(self, num_classes, pretrained=True, freeze_bn=False, **_):
        super(FCN8, self).__init__()
        vgg = models.vgg16_pth(pretrained)
        features = list(vgg.features.children())
        classifier = list(vgg.classifier.children())
        features[0].padding = 100, 100
        for layer in features:
            if 'MaxPool' in layer.__class__.__name__:
                layer.ceil_mode = True
        self.pool3 = nn.Sequential(*features[:17])
        self.pool4 = nn.Sequential(*features[17:24])
        self.pool5 = nn.Sequential(*features[24:])
        self.adj_pool3 = nn.Conv2D(256, num_classes, kernel_size=1)
        self.adj_pool4 = nn.Conv2D(512, num_classes, kernel_size=1)
        conv6 = nn.Conv2D(512, 4096, kernel_size=7)
        conv7 = nn.Conv2D(4096, 4096, kernel_size=1)
        output = nn.Conv2D(4096, num_classes, kernel_size=1)
        conv6.weight.data.copy_(classifier[0].weight.data.view(conv6.weight
            .data.size()))
        conv6.bias.data.copy_(classifier[0].bias.data)
        conv7.weight.data.copy_(classifier[3].weight.data.view(conv7.weight
            .data.size()))
        conv7.bias.data.copy_(classifier[3].bias.data)
        self.output = nn.Sequential(conv6, x2paddle.torch2paddle.ReLU(
            inplace=True), nn.Dropout(), conv7, x2paddle.torch2paddle.ReLU(
            inplace=True), nn.Dropout(), output)
        self.up_output = torch2paddle.Conv2DTranspose(num_classes,
            num_classes, kernel_size=4, stride=2, bias=False)
        self.up_pool4_out = torch2paddle.Conv2DTranspose(num_classes,
            num_classes, kernel_size=4, stride=2, bias=False)
        self.up_final = torch2paddle.Conv2DTranspose(num_classes,
            num_classes, kernel_size=16, stride=8, bias=False)
        self.up_output.weight.data.copy_(get_upsampling_weight(num_classes,
            num_classes, 4))
        self.up_pool4_out.weight.data.copy_(get_upsampling_weight(
            num_classes, num_classes, 4))
        self.up_final.weight.data.copy_(get_upsampling_weight(num_classes,
            num_classes, 16))
        for m in self.modules():
            if isinstance(m, torch2paddle.Conv2DTranspose):
                m.weight.requires_grad = False
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.pool3, self.pool4, self.pool5], False)

    def forward(self, x):
        imh_H, img_W = x.size()[2], x.size()[3]
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        output = self.output(pool5)
        up_output = self.up_output(output)
        adjstd_pool4 = self.adj_pool4(0.01 * pool4)
        add_out_pool4 = self.up_pool4_out(adjstd_pool4[:, :, 5:5 +
            up_output.size()[2], 5:5 + up_output.size()[3]] + up_output)
        adjstd_pool3 = self.adj_pool3(0.0001 * pool3)
        final_value = self.up_final(adjstd_pool3[:, :, 9:9 + add_out_pool4.
            size()[2], 9:9 + add_out_pool4.size()[3]] + add_out_pool4)
        final_value = final_value[:, :, 31:31 + imh_H, 31:31 + img_W
            ].contiguous()
        return final_value

    def get_backbone_params(self):
        return chain(self.pool3.parameters(), self.pool4.parameters(), self
            .pool5.parameters(), self.output.parameters())

    def get_decoder_params(self):
        return chain(self.up_output.parameters(), self.adj_pool4.parameters
            (), self.up_pool4_out.parameters(), self.adj_pool3.parameters(),
            self.up_final.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, paddle.nn.BatchNorm2D):
                module.eval()
