import torch
import numpy as np
import random

def get_binary_label(targets, index):
    ''' Cats have index 3, dogs have index 5 '''

    zeros = torch.zeros_like(targets)
    ones = torch.ones_like(targets)

    labels = torch.where(targets == index, ones, zeros)

    return labels


def create_unbalanced_CIFAR10(trainset, class_sizes = [625,625,625,5000,625,5000,625,625,625,625]):


  labels = np.array(trainset.targets)
  classes, sizes = np.unique(labels, return_counts=True)
  print(sizes)

  imbalanced_indices = []

  for i in range(len(classes)):
    indices = list(np.where(labels == i)[0])
    class_size = class_sizes[i]
    imbalanced_indices.extend(random.sample(indices, class_size))


  trainset.targets = labels[imbalanced_indices]
  trainset.data = trainset.data[imbalanced_indices]
  classes, sizes = np.unique(trainset.targets, return_counts=True)
  print(sizes)

  return trainset