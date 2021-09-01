import numpy as np
import time
import torch
from create_data import get_binary_label
from scheduler import congestion_avoid, linear_cong_condition, congestion_avoid_10classes
import copy


def test_congestion_avoider(start_time, testloader, device, model, optimizer, scheduler, branch_one_grads, branch_two_grads, branch_one_class, branch_two_class, branch_one_criterion, branch_two_criterion, epoch, max_epochs, min_cond, max_cond, min_epochs, mult):
    '''Same as original with additional function to increase the congestion condition linearly over the epochs'''

    model.eval()
    branch_one_test_loss = 0
    branch_two_test_loss = 0
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

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            branch_one_targets = get_binary_label(targets, index=branch_one_class)
            branch_two_targets = get_binary_label(targets, index=branch_two_class)
            inputs, branch_one_targets, branch_two_targets = inputs.to(device), branch_one_targets.to(device), branch_two_targets.to(device)
            branch_one_outputs, branch_two_outputs = model(inputs)
            branch_one_loss = branch_one_criterion(branch_one_outputs, branch_one_targets)
            branch_two_loss = branch_two_criterion(branch_two_outputs, branch_two_targets)

            branch_one_test_loss += branch_one_loss.item()
            branch_two_test_loss += branch_two_loss.item()
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

        branch_one_val_acc = 100.*branch_one_correct/branch_one_total
        
        if branch_one_TP + branch_one_FP > 0:
            branch_one_precision = 100.*branch_one_TP/(branch_one_TP + branch_one_FP)
        else:
            branch_one_precision = 0
        if branch_one_TP + branch_one_FN > 0:
            branch_one_recall = 100.*branch_one_TP/(branch_one_TP + branch_one_FN)
        else:
            branch_one_recall = 0
        
        branch_two_val_acc = 100.*branch_two_correct/branch_two_total
        
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

        condition = linear_cong_condition(min_cond, max_cond, epoch, max_epochs)

        optimizer, model, boolean_one, boolean_two, branch_one_grads, branch_two_grads = congestion_avoid(model, optimizer, branch_one_precision, branch_two_precision, condition, branch_one_grads, branch_two_grads, min_epochs, mult)
        scheduler.step()

        print("total test iters ", len(testloader), '| time: %.3f sec Cat Loss: %.3f | Cat Acc: %.3f%% (%d/%d) | Dog Loss: %.3f | Dog Acc: %.3f%% (%d/%d)'
        % ((time.time()-start_time), branch_one_test_loss/(batch_idx+1), 
        100.*branch_one_correct/branch_one_total, branch_one_correct, branch_one_total, 
        branch_two_test_loss/(batch_idx+1), 100.*branch_two_correct/branch_two_total, 
        branch_two_correct, branch_two_total))
        
        print('Cat P: : %.3f%% (%d/%d) | Dog P: %.3f%% (%d/%d)'%(branch_one_precision, branch_one_TP, branch_one_TP + branch_one_FP, branch_two_precision, branch_two_TP, branch_two_TP + branch_two_FP))
        print('Cat R: : %.3f%% (%d/%d) | Dog R: %.3f%% (%d/%d)'%(branch_one_recall, branch_one_TP, branch_one_TP + branch_one_FN, branch_two_recall, branch_two_TP, branch_two_TP + branch_two_FN))
        print('Cat F: : %.3f%%         | Dog R: %.3f%%'%(branch_one_F, branch_two_F))

    # RE-EVALUATE THE MODEL ON THE TEST SET AFTER THE WEIGHTS HAVE BEEN UPDATED
    model.eval()
    branch_one_test_loss = 0
    branch_two_test_loss = 0
    branch_one_correct = 0
    branch_two_correct = 0
    branch_one_total = 0
    branch_two_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            branch_one_targets = get_binary_label(targets, index=branch_one_class)
            branch_two_targets = get_binary_label(targets, index=branch_two_class)
            inputs, branch_one_targets, branch_two_targets = inputs.to(device), branch_one_targets.to(device), branch_two_targets.to(device)
            branch_one_outputs, branch_two_outputs = model(inputs)
            branch_one_loss = branch_one_criterion(branch_one_outputs, branch_one_targets)
            branch_two_loss = branch_two_criterion(branch_two_outputs, branch_two_targets)

            branch_one_test_loss += branch_one_loss.item()
            branch_two_test_loss += branch_two_loss.item()
            _, branch_one_predicted = branch_one_outputs.max(1)
            _, branch_two_predicted = branch_two_outputs.max(1)
            branch_one_total += branch_one_targets.size(0)
            branch_two_total += branch_two_targets.size(0)
            branch_one_correct += branch_one_predicted.eq(branch_one_targets).sum().item()
            branch_two_correct += branch_two_predicted.eq(branch_two_targets).sum().item()

        print("total test iters ", len(testloader), '| time: %.3f sec Cat Loss: %.3f | Cat Acc: %.3f%% (%d/%d) | Dog Loss: %.3f | Dog Acc: %.3f%% (%d/%d)'
        % ((time.time()-start_time), branch_one_test_loss/(batch_idx+1), 
        100.*branch_one_correct/branch_one_total, branch_one_correct, branch_one_total, 
        branch_two_test_loss/(batch_idx+1), 100.*branch_two_correct/branch_two_total, 
        branch_two_correct, branch_two_total))


    return optimizer, branch_one_val_acc, branch_two_val_acc, branch_one_precision, branch_two_precision, branch_one_recall, branch_two_recall, branch_one_F, branch_two_F, boolean_one, boolean_two, branch_one_grads, branch_two_grads


def test_congestion_avoider_10classes(cls_num, start_time, testloader, device, model, optimizer, scheduler, grads, criterion, epoch, max_epochs, min_cond, max_cond, min_epochs, mult, epoch_counts, num_class_avg, min_gradient):

    ''' 
        Inclusion of congestion avoidance strategy within the test function
    '''

    import copy

    print('Classes in test function: ', cls_num)

    model.eval()
    confusion_matrix = np.zeros((cls_num, cls_num))
    recalls = np.zeros((cls_num))
    precisions = np.zeros((cls_num))
    fScores = np.zeros((cls_num))
    boolean_values = np.zeros((cls_num))
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            _, predicted = outputs.max(1)

            for target, pred in zip(targets, predicted):
                confusion_matrix[target][pred] += 1
    
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

        condition = linear_cong_condition(min_cond, max_cond, epoch, max_epochs)
        
        optimizer, model, boolean_values, epoch_counts, grads = congestion_avoid_10classes(cls_num, model, optimizer, fScores, condition, grads, min_epochs, mult, epoch_counts, boolean_values, num_class_avg, min_gradient)
        scheduler.step()

    print('time: %.3f sec'% ((time.time()-start_time)))
    print('Rows: Actual, Columns: Predicted')
    print(confusion_matrix)
    for cls in range(cls_num):
        print('Class %d A: : %.3f%% (%d/%d)'%(cls, 100*accuracy, np.trace(confusion_matrix), confusion_matrix.sum()))
        print('Class %d P: : %.3f%% (%d/%d)'%(cls, 100*precisions[cls], confusion_matrix[cls][cls], confusion_matrix.sum(0)[cls]))
        print('Class %d R: : %.3f%% (%d/%d)'%(cls, 100*recalls[cls], confusion_matrix[cls][cls], confusion_matrix.sum(1)[cls]))
        print('Class %d F: : %.3f%%'%(cls, 100*fScores[cls]))
        print('********************')


    return optimizer, accuracy, precisions, recalls, fScores, boolean_values, grads, epoch_counts