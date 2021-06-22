import torch
import numpy as np

def get_binary_label(targets, index):
    ''' Cats have index 3, dogs have index 5 '''

    zeros = torch.zeros_like(targets)
    ones = torch.ones_like(targets)

    labels = torch.where(targets == index, ones, zeros)

    return labels


def create_unbalanced_CIFAR10(trainset, class_to_reduce, reduced_class_size):
    ''' Dummy code currently taken from here : 
    https://discuss.pytorch.org/t/how-to-implement-oversampling-in-cifar-10/16964/6 
    '''

    labels = np.array(trainset.targets)
    classes, count = np.unique(labels, return_counts=True)
    num_classes = len(classes)
    print(count)

    imbalanced_count = [5000 if i != class_to_reduce else reduced_class_size for i in range(num_classes) ]

    indices = [np.where(labels == i)[0] for i in range(num_classes)]

    imbalanced_indices = [idx[:class_count] for idx, class_count in zip(indices, imbalanced_count)]
    imbalanced_indices = np.hstack(imbalanced_indices)
    trainset.targets = labels[imbalanced_indices]
    trainset.data = trainset.data[imbalanced_indices]

    classes, count = np.unique(trainset.targets, return_counts=True)
    print(count)

    return trainset


def create_unbalanced_twoClass_CIFAR10(trainset, keep_classes=[3,5], new_count=[2000,5000]):
    ''' Dummy code currently taken from here : 
    https://discuss.pytorch.org/t/how-to-implement-oversampling-in-cifar-10/16964/6 
    '''

    labels = np.array(trainset.targets)
    classes, count = np.unique(labels, return_counts=True)
    num_classes = len(classes)
    print(count)

    indices = [np.where(labels == i)[0] for i in keep_classes]

    imbalanced_indices = [idx[:class_count] for idx, class_count in zip(indices, new_count)]
    imbalanced_indices = np.hstack(imbalanced_indices)
    trainset.targets = labels[imbalanced_indices]
    trainset.data = trainset.data[imbalanced_indices]

    classes, count = np.unique(trainset.targets, return_counts=True)
    print(count)

    return trainset