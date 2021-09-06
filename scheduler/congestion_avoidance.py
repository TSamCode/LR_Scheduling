import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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

    for cls in range(cls_num):
        metric = metrics[cls]
        if (threshold < condition * metric) and (epoch_counts[cls] >= min_epochs):
            boolean_values[cls] = True
            print('Condition has been met (class {}) ..... {:.2f}% --> {:.2f}%'.format(cls,100.*threshold, 100.*metric))
            for name, value in model.named_parameters():
                with torch.no_grad():
                    if name in grads[cls].keys() and (('layer4' in str(name)) or ('fc' in str(name))):
                            value += mult * grads[cls][name]
                            grads[cls][name] -= mult * grads[cls][name]
                            print('Gradient used: ', name)
            epoch_counts[cls] = 0
    
        else:
            print('Condition not met (class {}) ..... {:.2f}% --> {:.2f}%'.format(cls,100.*threshold, 100.*metric))

    return optimizer, model, boolean_values, epoch_counts, grads


def congestion_avoid_10classes_cosine(cls_num, model, optimizer, metrics, condition, grads, min_epochs, mult, epoch_counts, boolean_values, num_class_avg, similarity_threshold):

    # Create a threshold which is the average metric from a number of the worst performing classes
    threshold = np.average(np.sort(metrics)[:num_class_avg])

    for cls in range(cls_num):
        metric = metrics[cls]
        # If the metric of this class is sufficiently above the average of the worst classes
        # And the gradient of that layer is not significant compared to the other classes
        # then the knowledge from that class is returned
        if (threshold < condition * metric) and (epoch_counts[cls] >= min_epochs):
            boolean_values[cls] = True
            print('Condition has been met (class {}) ..... {:.2f}% --> {:.2f}%'.format(cls,100.*threshold, 100.*metric))
            for name, value in model.named_parameters():
                cosine_sims = [float(abs(F.cosine_similarity(grads[cls][name], grads[index][name], dim=-1)).mean()) for index in range(cls_num)]
                with torch.no_grad():
                    if name in grads[cls].keys():
                        if min(cosine_sims) < similarity_threshold:
                              value += mult * grads[cls][name]
                              grads[cls][name] -= mult * grads[cls][name]
                              print('Gradient used: ', name, ', Cosine sim: ', min(cosine_sims))
                        else:
                            print('Cosine similarity too high: ', name, '--> ', min(cosine_sims))
                            pass
            epoch_counts[cls] = 0
    
        else:
            print('Condition not met (class {}) ..... {:.2f}% --> {:.2f}%'.format(cls,100.*threshold, 100.*metric))

    return optimizer, model, boolean_values, epoch_counts, grads