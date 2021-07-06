from create_data import create_unbalanced_CIFAR10
import time
import torch

import torch.nn as nn
import torch.optim as optim
from training import train_branches, test_branches, train_TCP_branches, test_TCP_branches
from model import ResNetSplit18Shared, get_branch_params
from create_data import create_CIFAR_data

def get_results(model, trainloader, testloader, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, epochs, scheduler):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    if device == 'cuda':
        print('CUDA device used...')
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params",pytorch_total_params)
    print("pytorch_total_params_trainable",pytorch_total_params_trainable)

    branch_one_train_accuracies = []
    branch_two_train_accuracies = []
    branch_one_test_accuracies = []
    branch_two_test_accuracies = []

    start_time = time.time()

    for epoch in range(epochs):
        branch_one_train_acc, branch_two_train_acc = train_branches(device, model, trainloader, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, epoch, scheduler)
        branch_one_train_accuracies.append(branch_one_train_acc)
        branch_two_train_accuracies.append(branch_two_train_acc)

        branch_one_test_acc, branch_two_test_acc = test_branches(start_time, device, model, testloader, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, scheduler)
        branch_one_test_accuracies.append(branch_one_test_acc)
        branch_two_test_accuracies.append(branch_two_test_acc)

    return branch_one_train_accuracies, branch_two_train_accuracies, branch_one_test_accuracies, branch_two_test_accuracies


def get_congestion_results(branch_one_class=3, branch_two_class=5, epochs=100, min_epoch=10, decay_factor=0.02, mult_factor=2, condition=0.9, lr=0.1):

    trainset, trainloader, testset, testloader = create_CIFAR_data()
    trainset = create_unbalanced_CIFAR10(trainset)
    testset = create_unbalanced_CIFAR10(testset, [125,125,125,1000,125,1000,125,125,125,125])

    max_lr = lr

    model = ResNetSplit18Shared()
    branch_one_criterion = nn.CrossEntropyLoss()
    branch_two_criterion = nn.CrossEntropyLoss()    

    shared_params, branch1_params, branch2_params = get_branch_params(model)

    shared_optim = optim.SGD(shared_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    branch1_optim = optim.SGD(branch1_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    branch2_optim = optim.SGD(branch2_params, lr=lr, momentum=0.9, weight_decay=5e-4)

    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    if device == 'cuda':
        print('CUDA device used...')
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params",pytorch_total_params)
    print("pytorch_total_params_trainable",pytorch_total_params_trainable)

    branch_one_train_accuracies = []
    branch_two_train_accuracies = []
    branch_one_test_accuracies = []
    branch_two_test_accuracies = []

    prior_shared_params, prior_branch1_params, prior_branch2_params = get_branch_params(model)

    for epoch in range(epochs):
        branch_one_train_acc, branch_two_train_acc = train_TCP_branches(device, trainloader, min_epoch, max_lr, model, shared_optim, branch1_optim, branch2_optim, prior_shared_params, prior_branch1_params, prior_branch2_params, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, epoch, decay_factor, mult_factor, condition)
        branch_one_train_accuracies.append(branch_one_train_acc)
        branch_two_train_accuracies.append(branch_two_train_acc)

        branch_one_test_acc, branch_two_test_acc = test_TCP_branches(start_time, device, testloader, model, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class)
        branch_one_test_accuracies.append(branch_one_test_acc)
        branch_two_test_accuracies.append(branch_two_test_acc)

    return branch_one_train_accuracies, branch_two_train_accuracies, branch_one_test_accuracies, branch_two_test_accuracies


if __name__ == '__main__':
    get_congestion_results()