import time
import torch
from create_data import get_binary_label


def train_branches(device, model, trainloader, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, epoch, scheduler):

    print('\nEpoch: %d' % epoch)
    model.train()
    branch_one_train_loss = 0
    branch_two_train_loss = 0
    branch_one_correct = 0
    branch_two_correct = 0
    branch_one_total = 0
    branch_two_total = 0
    start_time = time.time()

    # The trainloader here needs to reference the imbalanced dataset (maybe only 2 classes)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print('Batch index: ', batch_idx)
        branch_one_targets = get_binary_label(targets, index=branch_one_class)
        branch_two_targets = get_binary_label(targets, index=branch_two_class)
        inputs, branch_one_targets, branch_two_targets = inputs.to(device), branch_one_targets.to(device), branch_two_targets.to(device)
        optimizer.zero_grad()
        # The outputs are now for the two separate branches
        branch_one_outputs, branch_two_outputs = model(inputs)
        branch_one_loss = branch_one_criterion(branch_one_outputs, branch_one_targets)
        branch_two_loss = branch_two_criterion(branch_two_outputs, branch_two_targets)
        loss = branch_one_loss + branch_two_loss
        loss.backward()
        #branch_one_loss.backward(retain_graph=True)
        #branch_two_loss.backward(retain_graph=True)
        optimizer.step()

        branch_one_train_loss += branch_one_loss.item()
        branch_two_train_loss += branch_two_loss.item()
        _, branch_one_predicted = branch_one_outputs.max(1)
        _, branch_two_predicted = branch_two_outputs.max(1)
        branch_one_total += branch_one_targets.size(0)
        branch_two_total += branch_two_targets.size(0)
        branch_one_correct += branch_one_predicted.eq(branch_one_targets).sum().item()
        branch_two_correct += branch_two_predicted.eq(branch_two_targets).sum().item()
    
    if scheduler is not None:
        scheduler.step()

    print("total train iters ", len(trainloader), '| time: %.3f sec Cat Loss: %.3f | Cat Acc: %.3f%% (%d/%d) | Dog Loss: %.3f | Dog Acc: %.3f%% (%d/%d)'
        % ((time.time()-start_time), branch_one_train_loss/(batch_idx+1), 
           100.*branch_one_correct/branch_one_total, branch_one_correct, branch_one_total, 
           branch_two_train_loss/(batch_idx+1), 100.*branch_two_correct/branch_two_total, 
           branch_two_correct, branch_two_total))

    return 100.*branch_one_correct/branch_one_total, 100.*branch_two_correct/branch_two_total