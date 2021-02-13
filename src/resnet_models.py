import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from src.resnet_blocks import SELayer, BasicBlock, SEBasicBlock, Bottleneck, SEBottleneck, Bottle2neck, SEBottle2neck
from src.pooling import StatsPooling

class ResNet(nn.Module):
    """ basic ResNet class: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py """
    def __init__(self, block, layers, num_classes, KaimingInit=False):
        
        self.inplanes = 16

        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

        self.classifier = nn.Linear(128 * block.expansion, num_classes)

        if KaimingInit == True:
            print('Using Kaiming Initialization.')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.size())

        x = self.layer1(x)
        #print(x.size())
        x = self.layer2(x)
        #print(x.size())
        x = self.layer3(x)
        #print(x.size())
        x = self.layer4(x)
        #print(x.size())

        x = self.avgpool(x).view(x.size()[0], -1)
        #print(x.shape)
        out = self.classifier(x)
        #print(out.shape)

        return F.log_softmax(out, dim=-1)

class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, m=0.35, num_classes=1000, loss='softmax', **kwargs):
        self.inplanes = 16
        super(Res2Net, self).__init__()
        self.loss = loss
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])#64
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)#128
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)#256
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)#512
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.stats_pooling = StatsPooling()

        if self.loss == 'softmax':
            # self.cls_layer = nn.Linear(2*8*128*block.expansion, num_classes)
            self.cls_layer = nn.Linear(128*block.expansion, num_classes)
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride,
                             stride=stride,
                             ceil_mode=True,
                             count_include_pad=False),
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample=downsample,
                  stype='stage',
                  baseWidth=self.baseWidth,
                  scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      baseWidth=self.baseWidth,
                      scale=self.scale))

        return nn.Sequential(*layers)

    def _forward(self, x):
        #x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # print('maxpool: ', x.size())

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())
        # x = self.stats_pooling(x)
        x = self.avgpool(x)
        # print('avgpool:', x.size())
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        x = self.cls_layer(x)

        return F.log_softmax(x, dim=-1)

    def extract(self, x):
        # x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        return x
    # Allow for accessing forward method in a inherited class
    forward = _forward

''' ResNet models'''
def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def se_resnet18(**kwargs):
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet34(**kwargs):
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet50(**kwargs):
    model = ResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)
    return model

'''Res2Net models'''
def res2net50_v1b(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)

    return model



def se_res2net50_v1b(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(SEBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    return model

def res2net50_v1b_14w_8s(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8, **kwargs)
    return model



def se_res2net50_v1b_14w_8s(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(SEBottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8, **kwargs)
    return model

def res2net50_v1b_26w_8s(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, **kwargs)
    return model



def se_res2net50_v1b_26w_8s(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(SEBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, **kwargs)
    return model


if __name__ == '__main__':
    images = torch.rand(2, 1, 257, 400)
    label = torch.randint(0, 2, (2,)).long()
    model = se_res2net50_v1b(pretrained=False, num_classes=3)
    #model = model.cuda(0)
    output = model(images)
    print(images.size())
    print(output.size())
