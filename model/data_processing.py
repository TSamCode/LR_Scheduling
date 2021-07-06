import torch
import numpy as np

def get_binary_label(targets, index):
    ''' Cats have index 3, dogs have index 5 '''

    zeros = torch.zeros_like(targets)
    ones = torch.ones_like(targets)

    labels = torch.where(targets == index, ones, zeros)

    return labels


def create_unbalanced_CIFAR10(trainset, class_sizes = [625,625,625,5000,625,5000,625,625,625,625]):

  labels = np.array(trainset.targets)
  classes, count = np.unique(labels, return_counts=True)
  num_classes = len(classes)
  print(count)

  indices = [np.where(labels == i)[0] for i in range(num_classes)]

  imbalanced_indices = [idx[:class_count] for idx, class_count in zip(indices, class_sizes)]
  imbalanced_indices = np.hstack(imbalanced_indices)
  trainset.targets = labels[imbalanced_indices]
  trainset.data = trainset.data[imbalanced_indices]

  classes, count = np.unique(trainset.targets, return_counts=True)
  print(count)

  return trainset