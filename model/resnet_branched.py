import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BasicBlock, Bottleneck, conv1x1

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

class ResNetSplit(nn.Module):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 10, zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64, replace_stride_with_dilation: Optional[List[bool]] = None, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(ResNetSplit, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        ##### BRANCH 1 LAYERS #####
        self.branch1conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.branch1bn1 = norm_layer(self.inplanes)
        self.branch1layer1 = self._make_layer(block, 64, layers[0])
        self.branch1layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.branch1layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.branch1layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.branch1fc = nn.Linear(512 * block.expansion, num_classes)


        ##### BRANCH 2 LAYERS #####
        self.branch2conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.branch2bn1 = norm_layer(self.inplanes)      
        self.branch2layer1 = self._make_layer(block, 64, layers[0])
        self.branch2layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.branch2layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.branch2layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])      
        self.branch2fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def _forward_branch_1(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        out = self.branch1conv1(x)
        out = self.branch1bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.branch1layer1(out)
        out = self.branch1layer2(out)
        out = self.branch1layer3(out)
        out = self.branch1layer4(out)

        out = self.avgpool(out)
        
        out = torch.flatten(out, 1)
        
        out = self.branch1fc(out)

        return out


    def _forward_branch_2(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        out = self.branch2conv1(x)
        out = self.branch2bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.branch2layer1(out)
        out = self.branch2layer2(out)
        out = self.branch2layer3(out)
        out = self.branch2layer4(out)

        out = self.avgpool(out)
        
        out = torch.flatten(out, 1)
        
        out = self.branch2fc(out)

        return out

    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_branch_1(x), self._forward_branch_2(x)


def ResNetSplit18():
    return ResNetSplit(BasicBlock, [2,2,2,2], num_classes=2)