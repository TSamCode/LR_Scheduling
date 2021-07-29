import torch
import numpy as np
import random

def get_binary_label(targets, index):
    ''' Cats have index 3, dogs have index 5 '''

    zeros = torch.zeros_like(targets)
    ones = torch.ones_like(targets)

    labels = torch.where(targets == index, ones, zeros)

    return labels


def get_branch_indices(targets, classes):

  bg = []
  indices = list(range(len(targets)))
  for index, target in enumerate(targets):
    if target not in classes:
      bg.append(index)

  branch_one_bg = random.sample(bg, int(len(bg) / 2))
  branch_two_bg = [x for x in bg if x not in branch_one_bg]

  branch_one_idx = [x for x in indices if x not in branch_two_bg]
  branch_two_idx = [x for x in indices if x not in branch_one_bg]

  return torch.tensor(branch_one_idx), torch.tensor(branch_two_idx)


def create_unbalanced_CIFAR10(trainset, class_sizes = [625,625,625,5000,625,5000,625,625,625,625]):
  '''
  Inspiration taken from PyTorch discussion blog: 
  https://discuss.pytorch.org/t/how-to-implement-oversampling-in-cifar-10/16964/6
  (Site accessed on 26th July 2021)
  '''

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