import torch.nn as nn
import torch.nn.functional as F
from model import BasicBlock, Bottleneck


class ResNetSplit(nn.Module):


    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNetSplit, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward_shared_block(self, x):
        return x


    def forward_branch_one(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def forward_branch_two(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def forward(self, x):
        x = self.forward_shared_block(x)
        branch_one = self.forward_branch_one(x)
        branch_two = self.forward_branch_two(x)
        return branch_one, branch_two


class ResNetSplit_TEST(nn.Module):
    ''' THIS METHOD DOES NOT SEEM TO WORK CURRENTLY! DIMENSION ERRORS OCCURRING'''

    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNetSplit, self).__init__()
        self.in_planes = 64

        self.branch1conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.branch1bn1 = nn.BatchNorm2d(64)
        self.branch1layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.branch1layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.branch1layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.branch1layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.branch1linear = nn.Linear(512*block.expansion, num_classes)

        self.branch2conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.branch2bn1 = nn.BatchNorm2d(64)
        self.branch2layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.branch2layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.branch2layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.branch2layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.branch2linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_shared_block(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        #return out
        return x
    
    def forward_branch_one(self, x):
        out = F.relu(self.branch1bn1(self.branch1conv1(x)))
        out = self.branch1layer1(out)
        out = self.branch1layer2(out)
        out = self.branch1layer3(out)
        out = self.branch1layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.branch1linear(out)
        return out

    def forward_branch_two(self, x):
        out = F.relu(self.branch2bn1(self.branch2conv1(x)))
        out = self.branch2layer1(out)
        out = self.branch2layer2(out)
        out = self.branch2layer3(out)
        out = self.branch2layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.branch2linear(out)
        return out

    def forward(self, x):
        #x = self.forward_shared_block(x)
        branch_one = self.forward_branch_one(x)
        branch_two = self.forward_branch_two(x)
        return branch_one, branch_two



def ResNetSplit18():
    return ResNetSplit(BasicBlock, [2,2,2,2])