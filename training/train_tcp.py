import torch
from create_data import get_binary_label
from scheduler import congestion_scheduler
import time

def train_TCP_branches(device, trainloader, min_epoch, model, shared_optim, branch1_optim, branch2_optim, prior_shared_params, prior_branch1_params, prior_branch2_params, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, epoch, decay_factor, mult_factor, condition):
    ''' 
        model = The model to be trained
        shared_optim, branch1_optim, branch2_optim = the optimizers used to determine how network weights are updated in each section of the network (e.g. SGD)
        prior_shared_params, prior_branch1_params, prior_branch2_params = The network parameters from the previous epoch, used by 'congestion_scheduler' to roll back the weights by one epoch
        branch_x_criterion = The criterion used to define the loss function
        branch_classes = Must be a list of length 2. Defines the classes that each branch of the model is learning to classify
        epoch = The current epoch in training
     '''

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
        
        branch_one_targets = get_binary_label(targets, index=branch_one_class)
        branch_two_targets = get_binary_label(targets, index=branch_two_class)
        inputs, branch_one_targets, branch_two_targets = inputs.to(device), branch_one_targets.to(device), branch_two_targets.to(device)
        shared_optim.zero_grad()
        branch1_optim.zero_grad()
        branch2_optim.zero_grad()
        # The outputs are now for the two separate branches
        branch_one_outputs, branch_two_outputs = model(inputs)
        branch_one_loss = branch_one_criterion(branch_one_outputs, branch_one_targets)
        branch_two_loss = branch_two_criterion(branch_two_outputs, branch_two_targets)
        loss = branch_one_loss + branch_two_loss
        loss.backward()

        branch_one_train_loss += branch_one_loss.item()
        branch_two_train_loss += branch_two_loss.item()
        _, branch_one_predicted = branch_one_outputs.max(1)
        _, branch_two_predicted = branch_two_outputs.max(1)
        branch_one_total += branch_one_targets.size(0)
        branch_two_total += branch_two_targets.size(0)
        branch_one_correct += branch_one_predicted.eq(branch_one_targets).sum().item()
        branch_two_correct += branch_two_predicted.eq(branch_two_targets).sum().item()

        shared_optim.step()
        branch1_optim.step()
        branch2_optim.step()

    branch1_acc = branch_one_correct/branch_one_total
    branch2_acc = branch_two_correct/branch_two_total
    print('Branch 1 training accuracy: ', 100.*branch1_acc)
    print('Branch 2 training accuracy: ', 100.*branch2_acc)
    shared_optim, branch1_optim, branch2_optim = congestion_scheduler(epoch, min_epoch, decay_factor, mult_factor, max_lr, shared_optim, branch1_optim, branch2_optim, prior_shared_params, prior_branch1_params, prior_branch2_params, branch1_acc, branch2_acc, condition)
    print('Shared LR: ', shared_optim.param_groups[0]['lr'])
    print('Branch 1 LR: ', branch1_optim.param_groups[0]['lr'])
    print('Branch 2 LR: ', branch2_optim.param_groups[0]['lr'])

    print("total train iters ", len(trainloader), '| time: %.3f sec Cat Loss: %.3f | Cat Acc: %.3f%% (%d/%d) | Dog Loss: %.3f | Dog Acc: %.3f%% (%d/%d)'
        % ((time.time()-start_time), branch_one_train_loss/(batch_idx+1), 
           100.*branch_one_correct/branch_one_total, branch_one_correct, branch_one_total, 
           branch_two_train_loss/(batch_idx+1), 100.*branch_two_correct/branch_two_total, 
           branch_two_correct, branch_two_total))

    return 100.*branch_one_correct/branch_one_total, 100.*branch_two_correct/branch_two_total