import time
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from create_data import get_binary_label, get_branch_indices
import copy


def train_congestion_avoider(trainloader, device, model, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, boolean_one, boolean_two, branch_one_grads, branch_two_grads, epoch_count_one, epoch_count_two):

    '''
    A function to train the branched version of the ResNet model learning to classify two image classes.

    Inputs:
        trainloader: The PyTroch data loader for training data
        device: string - The device the code is being run on
        model: The PyTorch branched ResNet model being trained
        optimizer: The model optimizer
        branch_x_criterion: The criterion used to define the loss function on branch 'x'
        branch_x_class: int - Defines the class that branch 'x' of the model is learning to classify
        boolean_x: boolean - indicator showing if a congestion event occurred on branch 'x'
        branch_x_grads: dict - The accumulated acquired knowledge for on branch 'x' of the network

    Returns:
        branch_x_acc: float - accuracy on the training data of branch 'x'
        branch_x_precision: float - precision on the training data of branch 'x'
        branch_x_recall: float - recall on the training data of branch 'x'
        branch_x_F: float - F-score on the training data of branch 'x'
        branch_x_grads: dict - The accumulated acquired knowledge for on branch 'x' of the network
     '''

    model.train()
    # Create variables for each branch that will be updated during training
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
    
    if boolean_two:
        epoch_count_one = 0
    if boolean_one:
        epoch_count_two = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Create binary training data labels for each branch of the model
        branch_one_targets = get_binary_label(targets, index=branch_one_class)
        branch_two_targets = get_binary_label(targets, index=branch_two_class)
        
        # Find the indices of the training data that will be passed to each branch of the model
        branch_one_idx, branch_two_idx = get_branch_indices(targets, classes=[branch_one_class, branch_two_class])
        branch_one_inputs = torch.index_select(inputs, 0, branch_one_idx)
        branch_one_targets = torch.index_select(branch_one_targets, 0, branch_one_idx)
        branch_two_inputs = torch.index_select(inputs, 0, branch_two_idx)
        branch_two_targets = torch.index_select(branch_two_targets, 0, branch_two_idx)
        
        # Move the tensors to the device (allows for CUDA functionality on Google Colab)
        branch_one_inputs, branch_two_inputs, branch_one_targets, branch_two_targets = branch_one_inputs.to(device), branch_two_inputs.to(device), branch_one_targets.to(device), branch_two_targets.to(device)
        optimizer.zero_grad()
        
        # For the training data passed to each branch of the model perform the forward pass
        branch_one_outputs, _ = model(branch_one_inputs)
        _, branch_two_outputs = model(branch_two_inputs)

        # Find the loss on each branch of the network
        branch_one_loss = branch_one_criterion(branch_one_outputs, branch_one_targets)
        branch_two_loss = branch_two_criterion(branch_two_outputs, branch_two_targets)
        
        # Back-propagate the loss due from branch one
        # Update the branch one acquired knowledge (as gradient multiplied by learning rate)
        # Save the knowledge in a temporary branch two dictionary
        branch_one_loss.backward(retain_graph=True)
        with torch.no_grad():
            # For each parameter update the acquired knowledge
            for name, parameter in model.named_parameters():
                try:
                    # The gradient is temporarily stored so that the difference can be determined on the second backward pass
                    branch_two_grads_tmp[name] = torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                    if name not in branch_one_grads.keys():
                        branch_one_grads[name] = torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                    else:
                        branch_one_grads[name] += torch.mul(copy.deepcopy(parameter.grad), optimizer.param_groups[0]['lr'])
                except:
                    pass

        # Back-propagate the loss due from branch two
        # Update the branch two acquired knowledge (as gradient multiplied by learning rate)
        # by observing the change in gradient from the results back-propagated from branch one
        branch_two_loss.backward(retain_graph=True)
        with torch.no_grad():
            # For each parameter update the acquired knowledge
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

        # The loss from each branch is summed and back-propagated. The model parameters updated by calling step()
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

        # Calculate the values in a confusion matrix to then calculate accuracy, precision, recall & F-score
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

    return branch_one_acc, branch_two_acc, branch_one_precision, branch_two_precision, branch_one_recall, branch_two_recall, branch_one_F, branch_two_F, branch_one_grads, branch_two_grads, epoch_count_one, epoch_count_two


def train_congestion_avoider_10_classes(device, model, trainloader, criterion, optimizer, cls_num, epoch_counts, grads):

  '''
  A function to train a ResNet model on the CIFAR-10 dataset with ten classes.
  Calculate the values of the confusion matrix and subsequent metrics.

  Inputs:
    device: string - The device the code is being run on
    model: The PyTorch ResNet model being trained
    trainloader: The PyTroch data loader for training data
    criterion: The loss criterion for the model
    optimizer: The model optimizer
    cls_num: int - The number of classes of images being trained
    epoch_counts: list - The number of epochs trained since the last congestion event
    boolean_values:
    grads: dict - The accumulated acquired knowledge for each image class & for each model parameter

  Returns:
    confusion_matrix:
    accuracy:
    recalls:
    precisions:
    fScores:
    grads:
    epoch_counts: number of epochs trained since the last congestion event
  '''

  model.train()
  start_time = time.time()

  # Create a matrix to store the confusion matrix results from the training epoch
  confusion_matrix = np.zeros((cls_num, cls_num))
  
  for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.to(device)
    targets = targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs,targets)

    # Create a mask for each class of data
    masks = [targets == k for k in range(cls_num)]
    # Find the loss due to each class of image
    sub_losses = [loss[mask].mean() for mask in masks]
    
    for name,param in model.named_parameters():
        param.grad = None

    # For each class of image accumulate the acquired knowledge of the network from these images
    for cls, sub_loss in enumerate(sub_losses):
      sub_loss.backward(retain_graph=True)
      for name,param in model.named_parameters():
        if name in grads[cls].keys():
          grads[cls][name] += torch.mul(copy.deepcopy(param.grad), optimizer.param_groups[0]['lr'])
        else:
          grads[cls][name] = torch.mul(copy.deepcopy(param.grad), optimizer.param_groups[0]['lr'])
        param.grad=None
    
    # The mean of the total loss is back-propagated and weights updated using step()
    optimizer.zero_grad()
    loss.mean().backward()  
    optimizer.step()

    _, predicted = outputs.max(1)

    for target, pred in zip(targets, predicted):
        confusion_matrix[target][pred] += 1

  # Create numpy arrays to store the metrics for each class of images
  recalls = np.zeros((cls_num))
  precisions = np.zeros((cls_num))
  fScores = np.zeros((cls_num))

  epoch_counts = [x+1 for x in epoch_counts]

  for cls in range(cls_num):
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

  accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()

  return confusion_matrix, accuracy, recalls, precisions, fScores, grads, epoch_counts