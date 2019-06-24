##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
from resnet import resnet50, resnet101, resnet152
from modules import View, Normalize, Encoding
from utils import MINCDataset

__all__ = ['DeepTen', 'get_deepten', 'get_deepten_resnet50_minc']


class DeepTen(nn.Module):
    def __init__(self, nclass, backbone):
        super(DeepTen, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if self.backbone == 'resnet50':
            self.pretrained = resnet50(pretrained=True, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101(pretrained=True, dilated=False)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet152(pretrained=True, dilated=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))
        self.scalenum = 4
        n_codes = 32

        self.fc = nn.Sequential(
            Normalize(),
            nn.Linear(64*64, 128),
        )
        self.classifier = nn.Sequential(Normalize(), nn.Linear(128*self.scalenum, nclass))
        self.head1 = self.make_layer_head_customize(256, n_codes, 56)
        self.head2 = self.make_layer_head_customize(512, n_codes, 28)
        self.head3 = self.make_layer_head_customize(1024, n_codes, 14)
        self.head4 = self.make_layer_head_customize(2048, n_codes, 7)
        self.pool1 = self.make_layer_pooling_customize(256, 56)
        self.pool2 = self.make_layer_pooling_customize(512, 28)
        self.pool3 = self.make_layer_pooling_customize(1024, 14)
        self.pool4 = self.make_layer_pooling_customize(2048, 7)

    def make_layer_head_customize(self, nchannels, n_codes, imgres):
        layers = []
        nchannels_fewer = 128
        featuredim = 64
        layers.append(nn.Sequential(
            nn.Conv2d(nchannels, nchannels_fewer, 1),
            nn.BatchNorm2d(nchannels_fewer),
            Encoding(D=nchannels_fewer, K=n_codes),
            View(-1, nchannels_fewer * n_codes),
            Normalize(),
            nn.Linear(nchannels_fewer * n_codes, featuredim),
        ))
        return nn.Sequential(*layers)


    def make_layer_pooling_customize(self, nchannels, imgres):
        layers = []
        nchannels_fewer = 128
        featuredim = 64
        layers.append(nn.Sequential(
            nn.Conv2d(nchannels, nchannels_fewer, 1),
            nn.AvgPool2d(imgres),
            View(-1, nchannels_fewer),
            nn.Linear(nchannels_fewer, featuredim),
            nn.BatchNorm1d(featuredim),
        ))
        return nn.Sequential(*layers)



    def forward(self, x):
        _, _, h, w = x.size()
        outputs = []
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        for i in range(self.scalenum):
            if i == 0:
                x_tmp = self.pretrained.layer1(x)
                x1 = self.head1(x_tmp)
                x2 = self.pool1(x_tmp)
            elif i == 1:
                x_tmp = self.pretrained.layer2(x)
                x1 = self.head2(x_tmp)
                x2 = self.pool2(x_tmp)
            elif i == 2:
                x_tmp = self.pretrained.layer3(x)
                x1 = self.head3(x_tmp)
                x2 = self.pool3(x_tmp)
            else:
                x_tmp = self.pretrained.layer4(x)
                x1 = self.head4(x_tmp)
                x2 = self.pool4(x_tmp)
            x = x_tmp
            x1 = x1.unsqueeze(1).expand(x1.size(0), x2.size(1), x1.size(-1))
            x_tmp = x1 * x2.unsqueeze(-1)
            x_tmp = x_tmp.view(-1, x1.size(-1) * x2.size(1))
            x_tmp = self.fc(x_tmp)
            outputs.append(x_tmp)
        x = torch.cat(outputs, 1)
        x = self.classifier(x)
        return x

def get_deepten(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                root='~/.encoding/models', **kwargs):
    r"""DeepTen model from the paper `"Deep TEN: Texture Encoding Network"
    <https://arxiv.org/pdf/1612.02844v1.pdf>`_
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_deepten(dataset='minc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    model = DeepTen(MINCDataset.NUM_CLASS, backbone=backbone, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('models/deepten_resnet50_minc-1225f149.pth'))
    return model

def get_deepten_resnet50_minc(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepTen model from the paper `"Deep TEN: Texture Encoding Network"
    <https://arxiv.org/pdf/1612.02844v1.pdf>`_
    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deepten_resnet50_minc(pretrained=True)
    >>> print(model)
    """
    return get_deepten(dataset='minc', backbone='resnet50', pretrained=pretrained,
                       root=root, **kwargs)
