from x2paddle import torch2paddle
import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from sklearn.utils import class_weight
from utils.lovasz_losses import lovasz_softmax


def make_one_hot(labels, classes):
    one_hot = torch2paddle.create_float32_tensor(labels.size()[0], classes,
        labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()
    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    weights = np.ones(7)
    weights[classes] = cls_w
    return paddle.to_tensor(weights).float().cuda()


class CrossEntropyLoss2d(nn.Layer):

    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=\
            ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class DiceLoss(nn.Layer):

    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1]
            )
        output = F.softmax(output, axis=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - (2.0 * intersection + self.smooth) / (output_flat.sum() +
            target_flat.sum() + self.smooth)
        return loss


class FocalLoss(nn.Layer):

    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True
        ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=ignore_index,
            weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = paddle.exp(-logpt)
        loss = (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Layer):

    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight
        =None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=\
            reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class LovaszSoftmax(nn.Layer):

    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, axis=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss
