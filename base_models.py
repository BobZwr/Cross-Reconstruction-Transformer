import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class SSLDataSet(Dataset):
    def __init__(self, data):
        super(SSLDataSet, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float)

    def __len__(self):
        return self.data.shape[0]

class FTDataSet(Dataset):
    def __init__(self, data, label, multi_label=False):
        super(FTDataSet, self).__init__()
        self.data = data
        self.label = label 
        self.multi_label = multi_label

    def __getitem__(self, index):
        if self.multi_label:
            return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.float))
        else:
            return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return self.data.shape[0]
    
# Resnet 1d

def conv(in_planes, out_planes, stride=1, kernel_size=3):
    "convolution with padding 自动使用zeros进行padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size - 1) // 2, bias=False)

class ZeroPad1d(nn.Module):
    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1d(nn.Module):
    """Bottleneck for ResNet52 ..."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        kernel_size = 3
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet1d(nn.Module):
    '''1d adaptation of the torchvision resnet'''

    def __init__(self, block, layers, kernel_size=3, input_channels=12, inplanes=64,
                 fix_feature_dim=False, kernel_size_stem=None, stride_stem=2, pooling_stem=True,
                 stride=2):
        super(ResNet1d, self).__init__()

        self.inplanes = inplanes
        layers_tmp = []
        if kernel_size_stem is None:
            kernel_size_stem = kernel_size[0] if isinstance(kernel_size, list) else kernel_size

        # conv-bn-relu (basic feature extraction)
        layers_tmp.append(nn.Conv1d(input_channels, inplanes,
                                    kernel_size=kernel_size_stem,
                                    stride=stride_stem,
                                    padding=(kernel_size_stem - 1) // 2, bias=False))
        layers_tmp.append(nn.BatchNorm1d(inplanes))
        layers_tmp.append(nn.ReLU(inplace=True))

        if pooling_stem is True:
            layers_tmp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        for i, l in enumerate(layers):
            if i == 0:
                layers_tmp.append(self._make_block(block, inplanes, layers[0]))
            else:
                layers_tmp.append(
                    self._make_block(block, inplanes if fix_feature_dim else (2 ** i) * inplanes, layers[i],
                                     stride=stride))

        self.feature_extractor = nn.Sequential(*layers_tmp)

    def _make_block(self, block, planes, blocks, stride=1, kernel_size=3):
        down_sample = None

        # 注定会进行下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, down_sample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.feature_extractor(x)

def resnet1d14(inplanes, input_channels):
    return ResNet1d(BasicBlock1d, [2,2,2], inplanes=inplanes, input_channels=input_channels)

def resnet1d18(**kwargs):
    return ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)

# MLP
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, bn = True):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.hidden_channels = hidden_channels
        self.fc1 = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, self.n_classes)
        self.ac = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_channels)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.ac(hidden)
        hidden = self.bn(hidden)
        out = self.fc2(hidden)

        return out

# Time-steps features -> aggregated features
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, tensor):
        b = tensor.size(0)
        return tensor.reshape(b, -1)

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."

    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)

    def forward(self, x):
        """x is shaped of B, C, T"""
        return torch.cat([self.mp(x), self.ap(x), x[..., -1:]], 1)

def bn_drop_lin(n_in, n_out, bn, p, actn):
    "`n_in`->bn->dropout->linear(`n_in`,`n_out`)->`actn`"
    layers = list()

    if bn:
        layers.append(nn.BatchNorm1d(n_in))

    if p > 0.:
        layers.append(nn.Dropout(p=p))

    layers.append(nn.Linear(n_in, n_out))

    if actn is not None:
        layers.append(actn)

    return layers

def create_head1d(nf: int, nc: int, lin_ftrs=[512, ], dropout=0.5, bn: bool = True, act="relu"):
    lin_ftrs = [3 * nf] + lin_ftrs + [nc]

    activations = [nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    layers = [AdaptiveConcatPool1d(), Flatten()]

    for ni, no, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], activations):
        layers += bn_drop_lin(ni, no, bn, dropout, actn)

    layers += [nn.Sigmoid()]

    return nn.Sequential(*layers)
