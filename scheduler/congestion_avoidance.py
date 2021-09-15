import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def linear_cong_condition(min_cond, max_cond, epoch, max_epochs):

    '''
    A function that will linearly interpolate between two values.

    Inputs:
        min_cond: float - The lower bound that is to be interpolated
        max_cond: float - The upper bound that is to be interpolated
        epoch: int - the current epoch of training
        max_epochs: int - The total number of epochs during training

    Returns:
        condition: float - The value linearly interpolated between min_cond & max_cond
    '''

    condition = min_cond + (max_cond - min_cond) * (epoch / max_epochs)

    return condition


def congestion_avoid(model, optimizer, branch1_metric, branch2_metric, condition, branch_one_grads, branch_two_grads, min_epochs, mult, epoch_count_one, epoch_count_two):

    '''
    A function to determine if a congestion event has occurred.
    When a congestion event occurs the parameters of the model must 
    return a proportion of acquired knowledge. The dictionaries containing
    the acquired knowledge are updated to reflect the 'lost' knowledge.

    Inputs:
        model: The PyTorch ResNet18 model being trained
        branch1_metric: float - The value of the metric returned by branch one in the test function
        branch2_metric: float - The value of the metric returned by branch two in the test function
        condition: float - The congestion condition parameter value
        branch_one_grads: dict - Accumulated acquired knowledgefrom branch one for each model parameter 
        branch_two_grads: dict - Accumulated acquired knowledgefrom branch two for each model parameter 
        min_epochs: int - The minimum number of epochs that must pass between successive congestion events on a model branch
        mult: float - The proportion of acquired knowledge to be returned on congestion

    Returns:
        optimizer: 
        model: The ResNet18 model being trained after parameter values have returned knowledge as required
        boolean_one: boolean - Has a congestion event occurred on branch two
        boolean_two: boolean - Has a congestion event occurred on branch one
        branch_one_grads: dict - The accumulated acquired knowledge on branch one after any knowledge has been returned due to congestion
        branch_two_grads: dict - The accumulated acquired knowledge on branch two after any knowledge has been returned due to congestion
    '''

    boolean_one = False
    boolean_two = False

    # Determine if a congestion event has occurred on either network branch
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

    return optimizer, model, boolean_one, boolean_two, branch_one_grads, branch_two_grads, epoch_count_one, epoch_count_two


def congestion_avoid_10classes(cls_num, model, optimizer, metrics, condition, grads, min_epochs, mult, epoch_counts, boolean_values, num_class_avg, similarity_threshold):

    '''
    A function to determine if a congestion event has occurred.
    When a congestion event occurs the parameters of the model must 
    return a proportion of acquired knowledge. The dictionaries containing
    the acquired knowledge are updated to reflect the 'lost' knowledge.

    Inputs:
        cls_num: int - The number of classes of images being classified
        model: The PyTorch ResNet18 model being trained
        metrics: list - The values of metrics for each class in the mult-class classifier using the test data
        condition: float - The congestion condition parameter value
        grads: dict - Accumulated acquired knowledge for each model parameter due to images in each class
        min_epochs: int - The minimum number of epochs that must pass between successive congestion events on a model branch
        mult: float - The proportion of acquired knowledge to be returned on congestion
        epoch_counts: list - The number of epochs since the last congestion event for each class of images
        boolean_values: list of boolean values - True at index 'cls' if a congestion event occurs for class 'cls'
        num_class_avg: int - The number of metrics used to determine a congestion threshold value
        similarity_threshold: float - Only parameters with a lower minimum cosine similarity will return acquired knowledge on congestion

    Returns:
        optimizer: 
        model: The ResNet18 model being trained after parameter values have returned knowledge as required
        boolean_values: boolean - Has a congestion event occurred for each class of images
        epoch_counts: list - The number of epochs since the last congestion event for each class of images
        grads: dict - The accumulated acquired knowledge due to each class of images after any knowledge has been returned due to congestion
    '''

    # Create a threshold which is the average metric from a number of the worst performing classes
    threshold = np.average(np.sort(metrics)[:num_class_avg])

    for cls in range(cls_num):
        metric = metrics[cls]
        # If the metric of this class is sufficiently above the average of the worst classes
        # then the knowledge from that class is returned
        if (threshold < condition * metric) and (epoch_counts[cls] >= min_epochs):
            boolean_values[cls] = True
            print('Condition has been met (class {}) ..... {:.2f}% --> {:.2f}%'.format(cls,100.*threshold, 100.*metric))
            for name, value in model.named_parameters():
                # Calculate the average cosine similarity (for each parameter) between the acquired knowledge of that class against all other classes
                cosine_sims = [float(abs(F.cosine_similarity(grads[cls][name], grads[index][name], dim=-1)).mean()) for index in range(cls_num)]
                with torch.no_grad():
                    if name in grads[cls].keys():
                        # Acquired knowledge is returned only if the minimum cosine similarity is below the threshold
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