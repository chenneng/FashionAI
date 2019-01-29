import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#from resnet
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class fashionModel(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(fashionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #add fashion fc
        self.fc_coat_length_labels = nn.Linear(512 * block.expansion, 8)
        self.fc_pant_length_labels = nn.Linear(512 * block.expansion, 6)
        self.fc_skirt_length_labels = nn.Linear(512 * block.expansion, 6)
        self.fc_sleeve_length_labels = nn.Linear(512 * block.expansion, 9)
        self.fc_collar_design_labels = nn.Linear(512 * block.expansion, 5)
        self.fc_lapel_design_labels = nn.Linear(512 * block.expansion, 5)
        self.fc_neck_design_labels = nn.Linear(512 * block.expansion, 5)
        self.fc_neckline_design_labels = nn.Linear(512 * block.expansion, 10)
        #self.fc = nn.Linear(512 * block.expansion, number_attr)
        nn.init.xavier_uniform(self.fc_coat_length_labels.weight)
        nn.init.constant(self.fc_coat_length_labels.bias, 0)
        nn.init.xavier_uniform(self.fc_pant_length_labels.weight)
        nn.init.constant(self.fc_pant_length_labels.bias, 0)
        nn.init.xavier_uniform(self.fc_skirt_length_labels.weight)
        nn.init.constant(self.fc_skirt_length_labels.bias, 0)
        nn.init.xavier_uniform(self.fc_sleeve_length_labels.weight)
        nn.init.constant(self.fc_sleeve_length_labels.bias, 0)
        nn.init.xavier_uniform(self.fc_collar_design_labels.weight)
        nn.init.constant(self.fc_collar_design_labels.bias, 0)
        nn.init.xavier_uniform(self.fc_lapel_design_labels.weight)
        nn.init.constant(self.fc_lapel_design_labels.bias, 0)
        nn.init.xavier_uniform(self.fc_neck_design_labels.weight)
        nn.init.constant(self.fc_neck_design_labels.bias, 0)
        nn.init.xavier_uniform(self.fc_neckline_design_labels.weight)
        nn.init.constant(self.fc_neckline_design_labels.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        #add fashion fc
        coat_length_labels = self.fc_coat_length_labels(x)
        pant_length_labels = self.fc_pant_length_labels(x)
        skirt_length_labels = self.fc_skirt_length_labels(x)
        sleeve_length_labels = self.fc_sleeve_length_labels(x)
        collar_design_labels = self.fc_collar_design_labels(x)
        lapel_design_labels = self.fc_lapel_design_labels(x)
        neck_design_labels = self.fc_neck_design_labels(x)
        neckline_design_labels = self.fc_neckline_design_labels(x)

        result = torch.cat((coat_length_labels, pant_length_labels, skirt_length_labels, sleeve_length_labels,
                                collar_design_labels, lapel_design_labels, neck_design_labels, neckline_design_labels), 1)

        return result

def creat_model():
    resnet50 = models.resnet50(pretrained = True)
    fashionNet = fashionModel(Bottleneck, [3, 4, 6, 3])
    pretrained_dict = resnet50.state_dict()
    model_dict = fashionNet.state_dict()
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    fashionNet.load_state_dict(model_dict)

    return fashionNet