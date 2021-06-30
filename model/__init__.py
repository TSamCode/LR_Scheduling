from model.cnn_blocks import BasicBlock, Bottleneck, conv1x1, conv3x3
from model.resnet import ResNet
from model.resnet_branched import ResNetSplit, ResNetSplit18
from model.resnet_branched_sharedLayers import ResNetSplitShared, ResNetSplit18Shared
from model.processing import get_branch_params