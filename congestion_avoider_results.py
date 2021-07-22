from training import train_congestion_avoider, test_congestion_avoider, train_congestion_avoider_weights, test_congestion_avoider_weights
from create_data import create_unbalanced_CIFAR10
import time
import torch

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CyclicLR
from model import ResNetSplit18Shared
from create_data import create_CIFAR_data


def get_cong_avoidance_results(branch_one_class=3, branch_two_class=5, epochs=100, min_cond=0.95, max_cond = 0.99, mult_factor=1, lr=0.1, min_epochs = 5):

    '''Allow the congestion condition to change linearly over time '''

    branch_one_grads = {}
    branch_two_grads = {}
    epoch_count_one = 0
    epoch_count_two = 0

    # IMPORT DATA
    trainset, trainloader, testset, testloader = create_CIFAR_data()
    
    # CREATE DATASET WITH CLASS SIZES
    trainset = create_unbalanced_CIFAR10(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = create_unbalanced_CIFAR10(testset, [125,125,125,1000,125,1000,125,125,125,125])
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    # CREATE MODEL
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNetSplit18Shared()
    model = model.to(device)
    if device == 'cuda':
        print('CUDA device used...')
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
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
    branch_one_condition = []
    branch_two_condition = []

    boolean_one = False
    boolean_two = False

    for epoch in range(epochs):
        print('\n********** EPOCH {} **********'.format(epoch + 1))
        print('Learning rate: ', optimizer.param_groups[0]['lr'])
        branch_one_train_acc, branch_two_train_acc, branch_one_grads, branch_two_grads = train_congestion_avoider(trainloader, model, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, boolean_one, boolean_two)
        #print('\nBRANCH ONE GRADS:')
        #for key, value in branch_one_grads.items():
        #    print(key, torch.sum(value))
        #print('\nBRANCH TWO GRADS:')
        #for key, value in branch_two_grads.items():
        #    print(key, torch.sum(value))
        branch_one_train_accuracies.append(branch_one_train_acc)
        branch_two_train_accuracies.append(branch_two_train_acc)
        print('Weight after training (SHARED): ', torch.sum(model.module.conv1.weight))
        print('Weight after training (BRANCH 1): ', torch.sum(model.module.branch1layer3[0].conv1.weight))
        print('Weight after training (BRANCH 2): ', torch.sum(model.module.branch2layer3[0].conv1.weight))
        optimizer, branch_one_val_acc, branch_two_val_acc, boolean_one, boolean_two = test_congestion_avoider(start_time, testloader, device, model, optimizer, scheduler, branch_one_grads, branch_two_grads, branch_one_class, branch_two_class, branch_one_criterion, branch_two_criterion, epoch, epochs, min_cond, max_cond, min_epochs, mult_factor)
        print('Weight after scheduler (SHARED): ', torch.sum(model.module.conv1.weight))
        print('Weight after training (BRANCH 1): ', torch.sum(model.module.branch1layer3[0].conv1.weight))
        print('Weight after training (BRANCH 2): ', torch.sum(model.module.branch2layer3[0].conv1.weight))
        branch_one_test_accuracies.append(branch_one_val_acc)
        branch_two_test_accuracies.append(branch_two_val_acc)
        branch_one_condition.append(boolean_one)
        branch_two_condition.append(boolean_two)

    return branch_one_train_accuracies, branch_two_train_accuracies, branch_one_test_accuracies, branch_two_test_accuracies


def get_cong_avoidance_weight_results(branch_one_class=3, branch_two_class=5, epochs=100, min_cond=0.95, max_cond = 0.99, mult_factor=1, lr=0.1, min_epochs = 5):

    branch_one_weight_update = {}
    branch_two_weight_update = {}
    epoch_count_one = 0
    epoch_count_two = 0
    boolean_one = False
    boolean_two = False

    # IMPORT DATA
    trainset, trainloader, testset, testloader = create_CIFAR_data()
    
    # CREATE DATASET WITH CLASS SIZES
    trainset = create_unbalanced_CIFAR10(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = create_unbalanced_CIFAR10(testset, [125,125,125,1000,125,1000,125,125,125,125])
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    # CREATE MODEL
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNetSplit18Shared()
    model = model.to(device)
    if device == 'cuda':
        print('CUDA device used...')
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
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
    branch_one_condition = []
    branch_two_condition = []

    for epoch in range(epochs):
        print('\n********** EPOCH {} **********'.format(epoch + 1))
        print('Learning rate: ', optimizer.param_groups[0]['lr'])
        branch_one_train_acc, branch_two_train_acc, branch_one_weight_update, branch_two_weight_update = train_congestion_avoider_weights(trainloader, model, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, boolean_one, boolean_two)
        
        print('\nBRANCH ONE WEIGHT UPDATE (DURING TRAINING):')
        print('Conv1: ', torch.sum(branch_one_weight_update[model.module.conv1.weight]))
        print('Branch 1 Layer 3: ', torch.sum(branch_one_weight_update[model.module.branch1layer3[0].conv1.weight]))
        print('Branch 2 Layer 3: ', torch.sum(branch_one_weight_update[model.module.branch2layer3[0].conv1.weight]))
        print('\nBRANCH TWO WEIGHT UPDATE (DURING TRAINING):')
        print('Conv1: ', torch.sum(branch_one_weight_update[model.module.conv1.weight]))
        print('Branch 1 Layer 3: ', torch.sum(branch_one_weight_update[model.module.branch1layer3[0].conv1.weight]))
        print('Branch 2 Layer 3: ', torch.sum(branch_one_weight_update[model.module.branch2layer3[0].conv1.weight]))

        branch_one_train_accuracies.append(branch_one_train_acc)
        branch_two_train_accuracies.append(branch_two_train_acc)
        print('Weight after training (SHARED): ', torch.sum(model.module.conv1.weight))
        print('Weight after training (BRANCH 1): ', torch.sum(model.module.branch1layer3[0].conv1.weight))
        print('Weight after training (BRANCH 2): ', torch.sum(model.module.branch2layer3[0].conv1.weight))
        optimizer, branch_one_val_acc, branch_two_val_acc, boolean_one, boolean_two = test_congestion_avoider_weights(start_time, testloader, device, model, optimizer, scheduler, branch_one_weight_update, branch_two_weight_update, branch_one_class, branch_two_class, branch_one_criterion, branch_two_criterion, epoch, epochs, min_cond, max_cond, min_epochs, mult_factor)
        print('Weight after scheduler (SHARED): ', torch.sum(model.module.conv1.weight))
        print('Weight after scheduler (BRANCH 1): ', torch.sum(model.module.branch1layer3[0].conv1.weight))
        print('Weight after scheduler (BRANCH 2): ', torch.sum(model.module.branch2layer3[0].conv1.weight))
        branch_one_test_accuracies.append(branch_one_val_acc)
        branch_two_test_accuracies.append(branch_two_val_acc)
        branch_one_condition.append(boolean_one)
        branch_two_condition.append(boolean_two)

    return branch_one_train_accuracies, branch_two_train_accuracies, branch_one_test_accuracies, branch_two_test_accuracies, branch_one_condition, branch_two_condition