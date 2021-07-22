import time
import torch
from torch.optim.optimizer import Optimizer
from create_data import get_binary_label


def train_congestion_avoider(trainloader, device, model, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, boolean_one, boolean_two):

    global branch_one_grads
    global branch_two_grads
    global epoch_count_one
    global epoch_count_two

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
    
    #if (epoch % reset_epochs == 0) or boolean_two:
    if boolean_two:
        # SHOULD I RESET THE GRADIENTS HERE OR SHOULD IT ALWAYS BE A ROLLING SUM!!!!
        branch_one_grads = {}
        epoch_count_one = 0
        #lr_one_cumulative = 0
    #if (epoch % reset_epochs == 0) or boolean_one:
    if boolean_one:
        # SHOULD I RESET THE GRADIENTS HERE OR SHOULD IT ALWAYS BE A ROLLING SUM!!!!
        branch_two_grads = {}
        epoch_count_two = 0
        #lr_two_cumulative = 0
    # The trainloader here needs to reference the imbalanced dataset (maybe only 2 classes)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        branch_one_targets = get_binary_label(targets, index=branch_one_class)
        branch_two_targets = get_binary_label(targets, index=branch_two_class)
        inputs, branch_one_targets, branch_two_targets = inputs.to(device), branch_one_targets.to(device), branch_two_targets.to(device)
        optimizer.zero_grad()
        branch_one_outputs, branch_two_outputs = model(inputs)
        branch_one_loss = branch_one_criterion(branch_one_outputs, branch_one_targets)
        branch_two_loss = branch_two_criterion(branch_two_outputs, branch_two_targets)
        
        # Back-propagate the loss due to 'cats'
        branch_one_loss.backward(retain_graph=True)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                try:
                    if name not in branch_one_grads.keys():
                        branch_one_grads[name] = torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                    else:
                        branch_one_grads[name] += torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                except:
                    print('ERROR: ', name)
                    print('Grads: ', branch_one_grads[name])
                    print('Current grad: ', copy.deepcopy(parameter.grad))
        model.zero_grad()

        branch_two_loss.backward(retain_graph=True)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                try:
                    if name not in branch_two_grads.keys():
                        branch_two_grads[name] = torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                    else:
                        branch_two_grads[name] += torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                except:
                    print('ERROR: ', name)
                    print('Grads: ', branch_two_grads[name], )
                    print('Current grad: ', copy.deepcopy(parameter.grad))
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

    epoch_count_one += 1
    epoch_count_two += 1

    branch_one_acc = 100.*branch_one_correct/branch_one_total
    branch_two_acc = 100.*branch_two_correct/branch_two_total

    print("total train iters ", len(trainloader), '| time: %.3f sec Cat Loss: %.3f | Cat Acc: %.3f%% (%d/%d) | Dog Loss: %.3f | Dog Acc: %.3f%% (%d/%d)'
        % ((time.time()-start_time), branch_one_train_loss/(batch_idx+1), 
           branch_one_acc, branch_one_correct, branch_one_total, 
           branch_two_train_loss/(batch_idx+1), branch_two_acc, 
           branch_two_correct, branch_two_total))

    return branch_one_acc, branch_two_acc, branch_one_grads, branch_two_grads


def train_congestion_avoider_weights(trainloader, model, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, boolean_one, boolean_two):
    
    global branch_one_weight_update
    global branch_two_weight_update
    global epoch_count_one
    global epoch_count_two

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
    #if (epoch % reset_epochs == 0) or boolean_two:
    if boolean_two:
        # SHOULD I RESET THE WEIGHT UPDATES HERE OR SHOULD IT JUST BE A ROLLING SUM AT ALL TIMES!!!!
        branch_one_weight_update = {}
        epoch_count_one = 0
        #lr_one_cumulative = 0
    #if (epoch % reset_epochs == 0) or boolean_one:
    if boolean_one:
        # SHOULD I RESET THE WEIGHT UPDATES HERE OR SHOULD IT JUST BE A ROLLING SUM AT ALL TIMES!!!!
        branch_two_weight_update = {}
        epoch_count_two = 0
        #lr_two_cumulative = 0
    # The trainloader here needs to reference the imbalanced dataset (maybe only 2 classes)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        branch_one_targets = get_binary_label(targets, index=branch_one_class)
        branch_two_targets = get_binary_label(targets, index=branch_two_class)
        #inputs, branch_one_targets, branch_two_targets = inputs.to(device), branch_one_targets.to(device), branch_two_targets.to(device)
        optimizer.zero_grad()
        branch_one_outputs, branch_two_outputs = model(inputs)
        branch_one_loss = branch_one_criterion(branch_one_outputs, branch_one_targets)
        
        # Back-propagate the loss due to 'cats'
        prior_params = copy.deepcopy(dict(model.name_parameters()))
        prior_state_dict = copy.deepcopy(model.state_dict())
        branch_one_loss.backward(retain_graph=True)
        # Update the weights due to the loss on cat classifier
        optimizer.step()
        # Store the weight updates due to cat classifier
        for name, parameter in model.named_parameters():
            try:
                if name not in branch_one_weight_update.keys():
                    branch_one_weight_update[name] = (copy.deepcopy(parameter) - prior_params[name])
                else:
                    branch_one_weight_update[name] += (copy.deepcopy(parameter) - prior_params[name])
            except:
                pass
        # Re-load old model and reset gradients
        model.load_state_dict(prior_state_dict)
        model.zero_grad()
        optimizer.zero_grad()

        branch_one_outputs, branch_two_outputs = model(inputs)
        branch_two_loss = branch_two_criterion(branch_two_outputs, branch_two_targets)
        # Back-propagate the loss due to 'dogs'
        branch_two_loss.backward(retain_graph=True)
        # Update the weights due to the loss on dog classifier
        optimizer.step()
        # Store the weight updates due to dog classifier
        for name, parameter in model.named_parameters():
            try:
                if name not in branch_two_weight_update.keys():
                    branch_two_weight_update[name] = (copy.deepcopy(parameter) - prior_params[name])
                else:
                    branch_two_weight_update[name] += (copy.deepcopy(parameter) - prior_params[name])
            except:
                pass
        # Re-load old model and reset gradients
        model.load_state_dict(prior_state_dict)
        model.zero_grad()
        optimizer.zero_grad()

        branch_one_outputs, branch_two_outputs = model(inputs)
        branch_one_loss = branch_one_criterion(branch_one_outputs, branch_one_targets)
        branch_two_loss = branch_two_criterion(branch_two_outputs, branch_two_targets)
        # Update model weights due to total loss
        total_loss = branch_one_loss + branch_two_loss
        total_loss.backward()
        optimizer.step()

        branch_one_train_loss += branch_one_loss.item()
        branch_two_train_loss += branch_two_loss.item()
        _, branch_one_predicted = branch_one_outputs.max(1)
        _, branch_two_predicted = branch_two_outputs.max(1)
        branch_one_total += branch_one_targets.size(0)
        branch_two_total += branch_two_targets.size(0)
        branch_one_correct += branch_one_predicted.eq(branch_one_targets).sum().item()
        branch_two_correct += branch_two_predicted.eq(branch_two_targets).sum().item()

    epoch_count_one += 1
    epoch_count_two += 1
    
    branch_one_acc = 100.*branch_one_correct/branch_one_total
    branch_two_acc = 100.*branch_two_correct/branch_two_total

    #optim, model = congestion_schedule_rollback(model, optim, branch_one_acc, branch_two_acc, condition, branch_one_grads, branch_two_grads)
    #scheduler.step()

    print("total train iters ", len(trainloader), '| time: %.3f sec Cat Loss: %.3f | Cat Acc: %.3f%% (%d/%d) | Dog Loss: %.3f | Dog Acc: %.3f%% (%d/%d)'
        % ((time.time()-start_time), branch_one_train_loss/(batch_idx+1), 
           branch_one_acc, branch_one_correct, branch_one_total, 
           branch_two_train_loss/(batch_idx+1), branch_two_acc, 
           branch_two_correct, branch_two_total))

    return branch_one_acc, branch_two_acc, branch_one_weight_update, branch_two_weight_update