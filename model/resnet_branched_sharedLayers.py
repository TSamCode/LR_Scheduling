import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BasicBlock, Bottleneck, conv1x1

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

class ResNetSplitShared(nn.Module):
    '''
    Inspiration taken from PyTorch discussion forum:
    https://discuss.pytorch.org/t/how-to-train-the-network-with-multiple-branches/2152
    (website accessed 26th July 2021)
    '''

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 10, zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64, replace_stride_with_dilation: Optional[List[bool]] = None, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(ResNetSplitShared, self).__init__()
        
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
        
        ##### SHARED LAYERS #####
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.layer1 = self._make_shared_layer(block, 64, layers[0])
        self.layer2 = self._make_shared_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        
        ##### BRANCH 1 LAYERS #####
        self.branch1_inplanes = 128
        self.branch1layer3 = self._make_branch1_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.branch1layer4 = self._make_branch1_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.branch1fc = nn.Linear(512 * block.expansion, num_classes)


        ##### BRANCH 2 LAYERS #####
        self.branch2_inplanes = 128
        self.branch2layer3 = self._make_branch2_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.branch2layer4 = self._make_branch2_layer(block, 512, layers[3], stride=2,
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


    def _make_shared_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        '''
        A function to create a shared layer of the network
        '''

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


    def _make_branch1_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        '''
        A function to create a layer of the network for branch one
        '''

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.branch1_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.branch1_inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.branch1_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        
        self.branch1_inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.branch1_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def _make_branch2_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        '''
        A function to create a layer of the network for branch two
        '''

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.branch2_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.branch2_inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.branch2_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        
        self.branch2_inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.branch2_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def get_branch_params(self):

        self.shared_params = [
                        {'params': self.conv1.parameters()},
                        {'params': self.bn1.parameters()},
                         {'params': self.layer1.parameters()},
                         {'params': self.layer2.parameters()},
        ]
        self.branch1_params = [
                        {'params': self.branch1layer3.parameters()},
                        {'params': self.branch1layer4.parameters()},
                         {'params': self.branch1fc.parameters()},
        ]
        self.branch2_params = [
                        {'params': self.branch2layer3.parameters()},
                        {'params': self.branch2layer4.parameters()},
                         {'params': self.branch2fc.parameters()},
        ]

        return self.shared_params, self.branch1_params, self.branch2_params


    def _forward_shared_branch(self, x:Tensor) -> Tensor:
        '''
        A function to perform the forward pass through the shared layers of the network
        '''

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        
        return out


    def _forward_branch_1(self, shared_out: Tensor) -> Tensor:
        '''
        A function to perform the forward pass through the 
        layers of the network found only in branch one
        '''

        branch1_out = self.branch1layer3(shared_out)
        branch1_out = self.branch1layer4(branch1_out)
        branch1_out = self.avgpool(branch1_out)
        branch1_out = torch.flatten(branch1_out, 1)
        branch1_out = self.branch1fc(branch1_out)

        return branch1_out


    def _forward_branch_2(self, shared_out: Tensor) -> Tensor:
        '''
        A function to perform the forward pass through the 
        layers of the network found only in branch two
        '''

        branch2_out = self.branch2layer3(shared_out)
        branch2_out = self.branch2layer4(branch2_out)
        branch2_out = self.avgpool(branch2_out)
        branch2_out = torch.flatten(branch2_out, 1)
        branch2_out = self.branch2fc(branch2_out)

        return branch2_out
    
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        A function to perform the complete forward pass through both branches of the network
        '''

        shared = self._forward_shared_branch(x)
        branch_one_out = self._forward_branch_1(shared)
        branch_two_out = self._forward_branch_2(shared)

        return branch_one_out, branch_two_out


def ResNetSplit18Shared():
    '''
    A function to create the ResNet18 model with two parallel branches
    '''

    return ResNetSplitShared(BasicBlock, [2,2,2,2])