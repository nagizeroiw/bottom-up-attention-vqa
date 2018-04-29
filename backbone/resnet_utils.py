import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        # img [batch_size, 3, 299, 299]

        x = img
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        fc = x.mean(3).mean(2).squeeze()
        if att_size > 0:
            batch, num_hid, _, __ = x.size()
            att = F.adaptive_avg_pool2d(x,[att_size,att_size]).view(-1, num_hid, att_size * att_size).permute(0, 2, 1)
            return fc, att
        else:
            return fc
