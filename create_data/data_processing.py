import torch
import numpy as np

def get_binary_label(targets, index):
    ''' Cats have index 3, dogs have index 5 '''

    zeros = torch.zeros_like(targets)
    ones = torch.ones_like(targets)

    labels = torch.where(targets == index, ones, zeros)

    return labels
