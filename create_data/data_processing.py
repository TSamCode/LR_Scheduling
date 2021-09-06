import torch
import numpy as np
import random

def get_binary_label(targets, index):
    ''' A function to convert the CIFAR-10 target labels
        from values from 0-10 to values in {0,1}.

        Inputs:
          targets: The CIFAR-10 data labels
          index: The index of the positive class

        Returns:
          labels: A new tensor of data labels with each value being 0 or 1
    '''

    zeros = torch.zeros_like(targets)
    ones = torch.ones_like(targets)

    labels = torch.where(targets == index, ones, zeros)

    return labels


def get_branch_indices(targets, classes):
  '''
  A function that determines the indices of input data that will be passed
  to each branch in the ResNet model  with two parallel branches.
  All images of the two classes learning to be classified are passed to both branches,
  images from the other eight classes (called 'background' images) are
  randomly passed to either of the two branches

  Inputs:
    targets: The target labels for the input data
    classes: The two classes of images that are learning to be classified

  Returns:
    Tensors containing the indices of input data to be passed to each branch of the network
  '''

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
  A function to alter the CIFAR-10 dataset to have a specified number of instances
  in each class of data.
  
  Inspiration for this solution has been taken from a PyTorch discussion blog discussing oversampling in the CIFAR dataset: 
  https://discuss.pytorch.org/t/how-to-implement-oversampling-in-cifar-10/16964/6
  (Site accessed on 26th July 2021)

  Inputs:
    trainset: The CIFAR-10 training data
    class_sizes: List of integers that will determine the number of training data items from each class

  Returns:
    trainset: An amended CIFAR-10 training dataset with the specified number of data items in each class
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


def create_class_subsets(trainset, shuffle=True, batch_size=128):

    '''
    A function to create subsets of the CIFAR-10 data, with each subset containing the positive examples from one class only.

    Inputs:
      trainset: The complete dataset that will be split into the respective subsets
      shuffle: A boolean value parameter that determines if the PyTorch data loader for each data loader should be shuffled
      batch_size: Integer parameter determining the number of items in each batch of the data loaders

    Returns:
      trainsets: A list of the ten data subsets created
      trainloders: A list of ten PyTorch ten data loaders, one for each data subset created
    '''
    
    labels = np.array(trainset.targets)
    
    trainset_0 = torch.utils.data.Subset(trainset, list(np.where(labels == 0)[0]))
    trainset_1 = torch.utils.data.Subset(trainset, list(np.where(labels == 1)[0]))
    trainset_2 = torch.utils.data.Subset(trainset, list(np.where(labels == 2)[0]))
    trainset_3 = torch.utils.data.Subset(trainset, list(np.where(labels == 3)[0]))
    trainset_4 = torch.utils.data.Subset(trainset, list(np.where(labels == 4)[0]))
    trainset_5 = torch.utils.data.Subset(trainset, list(np.where(labels == 5)[0]))
    trainset_6 = torch.utils.data.Subset(trainset, list(np.where(labels == 6)[0]))
    trainset_7 = torch.utils.data.Subset(trainset, list(np.where(labels == 7)[0]))
    trainset_8 = torch.utils.data.Subset(trainset, list(np.where(labels == 8)[0]))
    trainset_9 = torch.utils.data.Subset(trainset, list(np.where(labels == 9)[0]))

    trainsets = [trainset_0, trainset_1, trainset_2, trainset_3, trainset_4, trainset_5, trainset_6, trainset_7, trainset_8, trainset_9]

    trainloader_0 = torch.utils.data.DataLoader(trainset_0, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    trainloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    trainloader_3 = torch.utils.data.DataLoader(trainset_3, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    trainloader_4 = torch.utils.data.DataLoader(trainset_4, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    trainloader_5 = torch.utils.data.DataLoader(trainset_5, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    trainloader_6 = torch.utils.data.DataLoader(trainset_6, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    trainloader_7 = torch.utils.data.DataLoader(trainset_7, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    trainloader_8 = torch.utils.data.DataLoader(trainset_8, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    trainloader_9 = torch.utils.data.DataLoader(trainset_9, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    trainloaders = [trainloader_0, trainloader_1, trainloader_2, trainloader_3, trainloader_4, trainloader_5, trainloader_6, trainloader_7, trainloader_8, trainloader_9]

    return trainsets, trainloaders