import time
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from create_data import get_binary_label, get_branch_indices


def train_congestion_avoider(trainloader, device, model, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, boolean_one, boolean_two, branch_one_grads, branch_two_grads):

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
    branch_one_TP = 0
    branch_one_FP = 0
    branch_one_TN = 0
    branch_one_FN = 0
    branch_two_TP = 0
    branch_two_FP = 0
    branch_two_TN = 0
    branch_two_FN = 0
    branch_two_grads_tmp = {}
    start_time = time.time()
    
    #if (epoch % reset_epochs == 0) or boolean_two:
    if boolean_two:
        # SHOULD I RESET THE GRADIENTS HERE OR SHOULD IT ALWAYS BE A ROLLING SUM!!!!
        #branch_one_grads = {}
        epoch_count_one = 0
    #if (epoch % reset_epochs == 0) or boolean_one:
    if boolean_one:
        # SHOULD I RESET THE GRADIENTS HERE OR SHOULD IT ALWAYS BE A ROLLING SUM!!!!
        #branch_two_grads = {}
        epoch_count_two = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        branch_one_targets = get_binary_label(targets, index=branch_one_class)
        branch_two_targets = get_binary_label(targets, index=branch_two_class)
        
        branch_one_idx, branch_two_idx = get_branch_indices(targets, classes=[branch_one_class, branch_two_class])
        branch_one_inputs = torch.index_select(inputs, 0, branch_one_idx)
        branch_one_targets = torch.index_select(branch_one_targets, 0, branch_one_idx)
        branch_two_inputs = torch.index_select(inputs, 0, branch_two_idx)
        branch_two_targets = torch.index_select(branch_two_targets, 0, branch_two_idx)
        
        branch_one_inputs, branch_two_inputs, branch_one_targets, branch_two_targets = branch_one_inputs.to(device), branch_two_inputs.to(device), branch_one_targets.to(device), branch_two_targets.to(device)
        optimizer.zero_grad()
        
        branch_one_outputs, _ = model(branch_one_inputs)
        _, branch_two_outputs = model(branch_two_inputs)

        branch_one_loss = branch_one_criterion(branch_one_outputs, branch_one_targets)
        branch_two_loss = branch_two_criterion(branch_two_outputs, branch_two_targets)
        
        # Back-propagate the loss due to 'cats'
        branch_one_loss.backward(retain_graph=True)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                try:
                    branch_two_grads_tmp[name] = torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                    if name not in branch_one_grads.keys():
                        branch_one_grads[name] = torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                    else:
                        branch_one_grads[name] += torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                except:
                    pass

        branch_two_loss.backward(retain_graph=True)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    try:
                        if name not in branch_two_grads.keys():
                            if name in branch_two_grads_tmp.keys():
                                branch_two_grads[name] = (torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr']) - branch_two_grads_tmp[name])
                            else:
                                branch_two_grads[name] = torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                        else:
                            if name in branch_two_grads_tmp.keys():
                                branch_two_grads[name] += (torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr']) - branch_two_grads_tmp[name])
                            else:
                                branch_two_grads[name] += torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                    except:
                        pass
        optimizer.zero_grad()

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

        for target, pred in zip(branch_one_targets, branch_one_predicted):
          if target == 0:
            if pred == 0:
              branch_one_TN += 1
            else:
              branch_one_FP += 1
          elif target == 1:
            if pred == 1:
              branch_one_TP += 1
            else:
              branch_one_FN += 1
        
        for target, pred in zip(branch_two_targets, branch_two_predicted):
          if target == 0:
            if pred == 0:
              branch_two_TN += 1
            else:
              branch_two_FP += 1
          elif target == 1:
            if pred == 1:
              branch_two_TP += 1
            else:
              branch_two_FN += 1

    epoch_count_one += 1
    epoch_count_two += 1

    branch_one_acc = 100.*branch_one_correct/branch_one_total
    if branch_one_TP + branch_one_FP > 0:
      branch_one_precision = 100.*branch_one_TP/(branch_one_TP + branch_one_FP)
    else:
      branch_one_precision = 0
    if branch_one_TP + branch_one_FN > 0:
      branch_one_recall = 100.*branch_one_TP/(branch_one_TP + branch_one_FN)
    else:
      branch_one_recall = 0
    
    branch_two_acc = 100.*branch_two_correct/branch_two_total
    if branch_two_TP + branch_two_FP > 0:
      branch_two_precision = 100.*branch_two_TP/(branch_two_TP + branch_two_FP)
    else:
      branch_two_precision = 0
    if branch_two_TP + branch_two_FN > 0:
      branch_two_recall = 100.*branch_two_TP/(branch_two_TP + branch_two_FN)
    else:
      branch_two_recall = 0

    try:
      branch_one_F = 2 * branch_one_precision * branch_one_recall / (branch_one_precision + branch_one_recall)
    except:
      branch_one_F = 0
    try:
      branch_two_F = 2 * branch_two_precision * branch_two_recall / (branch_two_precision + branch_two_recall)
    except:
      branch_two_F = 0

    print("total train iters ", len(trainloader), '| time: %.3f sec Cat Loss: %.3f | Cat Acc: %.3f%% (%d/%d) | Dog Loss: %.3f | Dog Acc: %.3f%% (%d/%d)'
        % ((time.time()-start_time), branch_one_train_loss/(batch_idx+1), 
           branch_one_acc, branch_one_correct, branch_one_total, 
           branch_two_train_loss/(batch_idx+1), branch_two_acc, 
           branch_two_correct, branch_two_total))
    print('Cat P: : %.3f%% (%d/%d) | Dog P: %.3f%% (%d/%d)'% (branch_one_precision, branch_one_TP, branch_one_TP + branch_one_FP, branch_two_precision, branch_two_TP, branch_two_TP + branch_two_FP))
    print('Cat R: : %.3f%% (%d/%d) | Dog R: %.3f%% (%d/%d)'% (branch_one_recall, branch_one_TP, branch_one_TP + branch_one_FN, branch_two_recall, branch_two_TP, branch_two_TP + branch_two_FN))
    print('Cat F: : %.3f%%         | Dog F: %.3f%%'% (branch_one_F, branch_two_F))

    return branch_one_acc, branch_two_acc, branch_one_precision, branch_two_precision, branch_one_recall, branch_two_recall, branch_one_F, branch_two_F, branch_one_grads, branch_two_grads


def train_congestion_avoider_10classes(trainloaders, device, model, optimizer, criterion, boolean_values, grads, epoch_counts):

    ''' 
        Function to train ResNet model on ten classes of images, each class of images is passed to the model in turn
    '''

    import copy

    model.train()
    start_time = time.time()

    cls_num = len(trainloaders)
    confusion_matrix = np.zeros((cls_num, cls_num))

    for epoch_count, boolean in zip(epoch_counts, boolean_values):
        if boolean:
            epoch_count = 0
    
    for cls_num, trainloader in enumerate(trainloaders):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
        
            outputs = model(inputs)
            loss = criterion(outputs,targets)
        
            # Back-propagate the loss due to 'cats'
            loss.backward(retain_graph=True)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                  try:
                      if name not in grads[cls_num].keys():
                          grads[cls_num][name] = torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                      else:
                          grads[cls_num][name] += torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                  except:
                      pass
            optimizer.step()

            _, predicted = outputs.max(1)

            for target, pred in zip(targets, predicted):
                confusion_matrix[target][pred] += 1

    accuracies = np.zeros((10))
    recalls = np.zeros((10))
    precisions = np.zeros((10))
    fScores = np.zeros((10))

    for epoch_count in epoch_counts:
      epoch_count += 1

    for cls in range(cls_num):
        if confusion_matrix.sum() != 0:
            accuracies[cls] = confusion_matrix[cls][cls] / confusion_matrix.sum() 
        else:
            accuracies[cls] = 0
        if confusion_matrix.sum(1)[cls] != 0:
            recalls[cls] = confusion_matrix[cls][cls] / confusion_matrix.sum(1)[cls]
        else:
            recalls[cls] = 0
        if confusion_matrix.sum(0)[cls] != 0:
            precisions[cls] = confusion_matrix[cls][cls] / confusion_matrix.sum(0)[cls]
        else:
            precisions[cls] = 0
        if (precisions[cls] + recalls[cls]) != 0:
            fScores[cls] = 2 * precisions[cls] * recalls[cls] / (precisions[cls] + recalls[cls])
        else:
            fScores[cls] = 0

    return confusion_matrix, accuracies, recalls, precisions, fScores, grads, epoch_counts