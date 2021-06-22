import time
import torch

import torch.nn as nn
import torch.optim as optim
from training import train_branches, test_branches
from create_data import create_CIFAR_data
from model import ResNetSplit18 

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

if __name__ == '__main__':
    
    trainset, trainloader, testset, testloader = create_CIFAR_data()

    lr = 0.1
    model = ResNetSplit18()
    branch_one_criterion = nn.CrossEntropyLoss()
    branch_two_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    train_cat, train_dog, test_cat, test_dog = get_results(model, trainloader, testloader, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class=3, branch_two_class=5, epochs=10, scheduler=None)