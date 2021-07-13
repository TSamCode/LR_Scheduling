from training import train_congestion_avoider, test_congestion_avoider
from create_data import create_unbalanced_CIFAR10
import time
import torch

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from model import ResNetSplit18Shared
from create_data import create_CIFAR_data


def get_cong_avoidance_results(branch_one_class=3, branch_two_class=5, epochs=100, condition=0.975, decay_factor=1, mult_factor=1, lr=0.1):

    # IMPORT DATA
    trainset, trainloader, testset, testloader = create_CIFAR_data()
    
    # CREATE DATASET WITH CLASS SIZES
    trainset = create_unbalanced_CIFAR10(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = create_unbalanced_CIFAR10(testset, [125,125,125,1000,125,1000,125,125,125,125])
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    # CREATE MODEL
    model = ResNetSplit18Shared()
    # CREATE LOSS OF EACH BRANCH
    branch_one_criterion = nn.CrossEntropyLoss()
    branch_two_criterion = nn.CrossEntropyLoss()
    # CREATE MODEL OPTIMIZER
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=5e-4)
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=lr, step_size_up=10, mode="triangular2")

    # BEGIN RECORDING THE TIME
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    branch_one_train_accuracies = []
    branch_two_train_accuracies = []
    branch_one_test_accuracies = []
    branch_two_test_accuracies = []

    for epoch in range(epochs):
        # TRAIN THE MODEL FOR ONE EPOCH
        branch_one_train_acc, branch_two_train_acc, branch_one_grads, branch_two_grads = train_congestion_avoider(trainloader, model, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class)
        # SAVE THE TRAINING ACCURACIES OF EACH BRANCH
        branch_one_train_accuracies.append(branch_one_train_acc)
        branch_two_train_accuracies.append(branch_two_train_acc)
        # TEST THE MODEL
        # USE THE ACCURACIES TO STEP THE OPTIMIZER & SCHEDULER
        optimizer, branch_one_val_acc, branch_two_val_acc = test_congestion_avoider(start_time, testloader, device, model, optimizer, scheduler, branch_one_grads, branch_two_grads, branch_one_class, branch_two_class, branch_one_criterion, branch_two_criterion, condition)
        # SAVE THE TEST ACCURACIES OF EACH BRANCH
        branch_one_test_accuracies.append(branch_one_val_acc)
        branch_two_test_accuracies.append(branch_two_val_acc)

    return branch_one_train_accuracies, branch_one_test_accuracies, branch_two_train_accuracies, branch_two_test_accuracies