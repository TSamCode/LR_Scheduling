import time
import torch
from create_data import get_binary_label

def train_congestion_avoider(trainloader, model, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class):
    
    ''' 
        model = The model to be trained
        shared_optim, branch1_optim, branch2_optim = the optimizers used to determine how network weights are updated in each section of the network (e.g. SGD)
        prior_shared_params, prior_branch1_params, prior_branch2_params = The network parameters from the previous epoch, used by 'congestion_scheduler' to roll back the weights by one epoch
        branch_x_criterion = The criterion used to define the loss function
        branch_classes = Must be a list of length 2. Defines the classes that each branch of the model is learning to classify
        epoch = The current epoch in training
     '''

    import copy

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
        #inputs, branch_one_targets, branch_two_targets = inputs.to(device), branch_one_targets.to(device), branch_two_targets.to(device)
        optimizer.zero_grad()
        branch_one_outputs, branch_two_outputs = model(inputs)
        branch_one_loss = branch_one_criterion(branch_one_outputs, branch_one_targets)
        branch_two_loss = branch_two_criterion(branch_two_outputs, branch_two_targets)
        
        # Back-propagate the loss due to 'cats'
        branch_one_loss.backward(retain_graph=True)
        branch_one_grads = {}
        for name, parameter in model.named_parameters():
            if name not in branch_one_grads.keys():
                branch_one_grads[name] = copy.deepcopy(parameter.grad)
            else:
                branch_one_grads[name] += copy.deepcopy(parameter.grad)
        model.zero_grad()

        branch_two_loss.backward(retain_graph=True)
        branch_two_grads = {}
        for name, parameter in model.named_parameters():
            if name not in branch_two_grads.keys():
                branch_two_grads[name] = copy.deepcopy(parameter.grad)
            else:
                branch_two_grads[name] += copy.deepcopy(parameter.grad)
        model.zero_grad()

        total_loss = branch_one_loss + branch_two_loss
        total_loss.backward()
        total_grads = {}
        for name, parameter in model.named_parameters():
            if name not in total_grads.keys():
                total_grads[name] = copy.deepcopy(parameter.grad)
            else:
                total_grads[name] += copy.deepcopy(parameter.grad)
        optimizer.step()

        branch_one_train_loss += branch_one_loss.item()
        branch_two_train_loss += branch_two_loss.item()
        _, branch_one_predicted = branch_one_outputs.max(1)
        _, branch_two_predicted = branch_two_outputs.max(1)
        branch_one_total += branch_one_targets.size(0)
        branch_two_total += branch_two_targets.size(0)
        branch_one_correct += branch_one_predicted.eq(branch_one_targets).sum().item()
        branch_two_correct += branch_two_predicted.eq(branch_two_targets).sum().item()

    branch_one_acc = 100.*branch_one_correct/branch_one_total
    branch_two_acc = 100.*branch_two_correct/branch_two_total

    #optim, model = congestion_schedule_rollback(model, optim, branch_one_acc, branch_two_acc, condition, branch_one_grads, branch_two_grads)
    #scheduler.step()

    print("total train iters ", len(trainloader), '| time: %.3f sec Cat Loss: %.3f | Cat Acc: %.3f%% (%d/%d) | Dog Loss: %.3f | Dog Acc: %.3f%% (%d/%d)'
        % ((time.time()-start_time), branch_one_train_loss/(batch_idx+1), 
           branch_one_acc, branch_one_correct, branch_one_total, 
           branch_two_train_loss/(batch_idx+1), branch_two_acc, 
           branch_two_correct, branch_two_total))

    return branch_one_acc, branch_two_acc, branch_one_grads, branch_two_grads