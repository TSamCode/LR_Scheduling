import torch
import torch.nn as nn
import numpy as np


def linear_cong_condition(min_cond, max_cond, epoch, max_epochs):

    condition = min_cond + (max_cond - min_cond) * (epoch / max_epochs)

    return condition


def congestion_avoid(model, optimizer, branch1_metric, branch2_metric, condition, branch_one_grads, branch_two_grads, min_epochs, mult):

    global epoch_count_one
    global epoch_count_two

    boolean_one = False
    boolean_two = False

    branch1_cond = (branch1_metric < condition * branch2_metric) and (epoch_count_two >= min_epochs)
    branch2_cond = (branch2_metric < condition * branch1_metric) and (epoch_count_one >= min_epochs)

    if branch1_cond:
        boolean_one = True
        print('Branch 1 condition has been met ..... {:.2f}%'.format(100.*condition))
        for name, value in model.named_parameters():
            with torch.no_grad():
                if name in branch_two_grads.keys():
                    value += mult * branch_two_grads[name]
        for name in branch_two_grads.keys():
            branch_two_grads[name] -= mult * branch_two_grads[name]
        epoch_count_two = 0

    elif branch2_cond:
        boolean_two = True
        print('Branch 2 condition has been met ..... {:.2f}%'.format(100.*condition))
        for name, value in model.named_parameters():
            with torch.no_grad():
                if name in branch_one_grads.keys():
                    value += mult * branch_one_grads[name]
        for name in branch_one_grads.keys():
            branch_one_grads[name] -= mult * branch_one_grads[name]
        epoch_count_one = 0
    
    else:
        print('No condition is met ..... {:.2f}%'.format(100.*condition))

    return optimizer, model, boolean_one, boolean_two, branch_one_grads, branch_two_grads


def congestion_avoid_10classes(cls_num, model, optimizer, metrics, condition, grads, min_epochs, mult, epoch_counts, boolean_values, num_class_avg, min_gradient):

    # Create a threshold which is the average metric from a number of the worst performing classes
    threshold = np.average(np.sort(metrics)[:num_class_avg])

    total_grad = {'layer1': {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 'total':0},
                  'layer2': {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 'total':0},
                  'layer3': {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 'total':0},
                  'layer4': {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 'total':0},
                  'conv1': {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 'total':0},
                  'bn1': {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 'total':0},
                  'fc': {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 'total':0}
                  }

    for cls in range(cls_num):
        for name, value in model.named_parameters():
            if 'layer1' in str(name):
                total_grad['layer1'][cls] += float(abs(grads[cls][name]).sum())
                total_grad['layer1']['total'] += float(abs(grads[cls][name]).sum())
            elif 'layer2' in str(name):
                total_grad['layer2'][cls] += float(abs(grads[cls][name]).sum())
                total_grad['layer2']['total'] += float(abs(grads[cls][name]).sum())
            elif 'layer3' in str(name):
                total_grad['layer3'][cls] += float(abs(grads[cls][name]).sum())
                total_grad['layer3']['total'] += float(abs(grads[cls][name]).sum())
            elif 'layer4' in str(name):
                total_grad['layer4'][cls] += float(abs(grads[cls][name]).sum())
                total_grad['layer4']['total'] += float(abs(grads[cls][name]).sum())
            elif 'conv1' in str(name):
                total_grad['conv1'][cls] += float(abs(grads[cls][name]).sum())
                total_grad['conv1']['total'] += float(abs(grads[cls][name]).sum())
            elif 'bn1' in str(name):
                total_grad['bn1'][cls] += float(abs(grads[cls][name]).sum())
                total_grad['bn1']['total'] += float(abs(grads[cls][name]).sum())
            elif 'fc' in str(name):
                total_grad['fc'][cls] += float(abs(grads[cls][name]).sum())
                total_grad['fc']['total'] += float(abs(grads[cls][name]).sum())

    for cls in range(cls_num):
        metric = metrics[cls]
        # If the metric of this class is sufficiently above the average of the worst classes
        # And the gradient of that layer is not significant compared to the other classes
        # then the knowledge from that class is returned
        if (threshold < condition * metric) and (epoch_counts[cls] >= min_epochs):
            boolean_values[cls] = True
            print('Condition has been met (class {}) ..... {:.2f}% --> {:.2f}%'.format(cls,100.*threshold, 100.*metric))
            for name, value in model.named_parameters():
                with torch.no_grad():
                  # ADDED CONDITION TO ONLY CHANGE GRADIENTS FROM LAYER 3 ONWARDS
                    if name in grads[cls].keys():# and (('layer4' in str(name)) or ('fc' in str(name))):
                        if ('layer1' in str(name)) and (total_grad['layer1'][cls] < min_gradient * total_grad['layer1']['total']):
                            value += mult * grads[cls][name]
                            grads[cls][name] -= mult * grads[cls][name]
                        elif ('layer2' in str(name)) and (total_grad['layer2'][cls] < min_gradient * total_grad['layer2']['total']):
                            value += mult * grads[cls][name]
                            grads[cls][name] -= mult * grads[cls][name]
                        elif ('layer3' in str(name)) and (total_grad['layer3'][cls] < min_gradient * total_grad['layer3']['total']):
                            value += mult * grads[cls][name]
                            grads[cls][name] -= mult * grads[cls][name]
                        elif ('layer4' in str(name)) and (total_grad['layer4'][cls] < min_gradient * total_grad['layer4']['total']):
                            value += mult * grads[cls][name]
                            grads[cls][name] -= mult * grads[cls][name]
                        elif ('conv1' in str(name)) and (not any(layer in str(name) for layer in ('layer1','layer2','layer3','layer4'))) and (total_grad['conv1'][cls] < min_gradient * total_grad['conv1']['total']):
                            value += mult * grads[cls][name]
                            grads[cls][name] -= mult * grads[cls][name]
                        elif ('bn1' in str(name)) and (not any(layer in str(name) for layer in ('layer1','layer2','layer3','layer4'))) and (total_grad['bn1'][cls] < min_gradient * total_grad['bn1']['total']):
                            value += mult * grads[cls][name]
                            grads[cls][name] -= mult * grads[cls][name]
                        elif ('fc' in str(name)) and (not any(layer in str(name) for layer in ('layer1','layer2','layer3','layer4'))) and (total_grad['fc'][cls] < min_gradient * total_grad['fc']['total']):
                            value += mult * grads[cls][name]
                            grads[cls][name] -= mult * grads[cls][name]
                        else:
                            print('Gradient not used: ', name)
                            pass
            epoch_counts[cls] = 0
    
        else:
            print('Condition not met (class {}) ..... {:.2f}% --> {:.2f}%'.format(cls,100.*threshold, 100.*metric))

    return optimizer, model, boolean_values, epoch_counts, grads