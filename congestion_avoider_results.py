from training import train_congestion_avoider, test_congestion_avoider, train_congestion_avoider_10_classes, test_congestion_avoider_10classes
from create_data import create_unbalanced_CIFAR10, create_CIFAR_data, create_sampled_CIFAR10_data
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CyclicLR
from model import ResNet18, ResNetSplit18, ResNetSplit18Shared
from utils import save_to_file, plot_results_multiClass, plot_results_diff_multiClass, plot_results_twoClass, plot_results_diff_twoClass


def get_cong_avoidance_results(branch_one_class=5, branch_two_class=9, class_sizes_train = [625,625,625,625,625,1000,625,625,625,5000], class_sizes_test = [125,125,125,125,125,1000,125,125,125,1000], epochs=100, min_cond=0.95, max_cond = 0.99, mult_factor=1, lr=0.1, min_epochs = 5, metric='recall'):

    '''
    A function to produce the results of training the branched ResNet model 
    using the congestion avoidance scheduler.

    Inputs:
      branch_x_class: int - The class number of image that branch 'x' is learning to classify
      class_sizes_train: list - Number of training inputs for each class of images
      class_sizes_test: list - Number of test data inputs for each class of images
      epochs: int - the number of epochs the model is trained for
      min_cond: float - the lower bound of the congestion condition parameter that will be interpolated between
      max_cond: float - the upper bound of the congestion condition parameter that will be interpolated between
      mult_factor: float - the multiplicative decrease factor that determines the proportion of knowledge returned
      lr: float - The initial learning rate
      min_epochs: int - the minimum number of epochs that must pass between successive congestion events
      metric: ['recall','precision','F-score'] - parameter to determine which metric is used to determine a congestion event

    Returns:
      branch_x_train_accuracies: list - the accuracy of the model on the training data at each epoch
      branch_x_train_P: list - the precision of the model on the training data at each epoch
      branch_x_train_R: list - the recall of the model on the training data at each epoch
      branch_x_train_F: list - the F-score of the model on the training data at each epoch
      branch_x_test_accuracies: list - the accuracy of the model on the test data at each epoch
      branch_x_test_P: list - the precision of the model on the test data at each epoch
      branch_x_test_R: list - the recall of the model on the test data at each epoch
      branch_x_test_F: list - the F-score of the model on the test data at each epoch
      branch_x_condition: list - indicator to show the epochs at which congestion events occurred on each branch
    '''

    branch_one_grads = {}
    branch_two_grads = {}
    epoch_count_one = 0
    epoch_count_two = 0

    # IMPORT DATA
    trainset, trainloader, testset, testloader = create_CIFAR_data()
    
    # CREATE DATASET WITH CLASS SIZES (NOW CAT DATA IS 10x SMALLER)
    trainset = create_unbalanced_CIFAR10(trainset, class_sizes = class_sizes_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = create_unbalanced_CIFAR10(testset, class_sizes = class_sizes_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    # CREATE MODEL
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNetSplit18Shared(num_classes=2)
    model = model.to(device)
    if device == 'cuda':
        print('CUDA device used...')
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    # CREATE LOSS OF EACH BRANCH
    branch_one_samples = [class_sizes_train[branch_two_class] + 5000 / 2, class_sizes_train[branch_one_class]]
    branch_one_weights = torch.tensor([(sum(branch_one_samples) - x)/sum(branch_one_samples) for x in branch_one_samples])
    branch_two_samples = [class_sizes_train[branch_one_class] + 5000 / 2, class_sizes_train[branch_two_class]]
    branch_two_weights = torch.tensor([(sum(branch_two_samples) - x)/sum(branch_two_samples) for x in branch_two_samples])
    branch_one_weights, branch_two_weights = branch_one_weights.to(device), branch_two_weights.to(device)
    
    branch_one_criterion = nn.CrossEntropyLoss(weight=branch_one_weights)
    branch_two_criterion = nn.CrossEntropyLoss(weight=branch_two_weights)
    # CREATE MODEL OPTIMIZER
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=5e-4)
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=lr, step_size_up=10, mode="triangular2")

    # BEGIN RECORDING THE TIME
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    branch_one_train_accuracies = []
    branch_two_train_accuracies = []
    branch_one_train_P = []
    branch_two_train_P = []
    branch_one_train_R = []
    branch_two_train_R = []
    branch_one_train_F = []
    branch_two_train_F = []

    branch_one_test_accuracies = []
    branch_two_test_accuracies = []
    branch_one_test_P = []
    branch_two_test_P = []
    branch_one_test_R = []
    branch_two_test_R = []
    branch_one_test_F = []
    branch_two_test_F = []

    branch_one_condition = []
    branch_two_condition = []

    boolean_one = False
    boolean_two = False

    for epoch in range(epochs):
        print('\n********** EPOCH {} **********'.format(epoch + 1))
        print('Learning rate: ', optimizer.param_groups[0]['lr'])
        branch_one_train_acc, branch_two_train_acc, b1_train_P, b2_train_P, b1_train_R, b2_train_R, b1_train_F, b2_train_F, branch_one_grads, branch_two_grads, epoch_count_one, epoch_count_two = train_congestion_avoider(trainloader, device, model, optimizer, branch_one_criterion, branch_two_criterion, branch_one_class, branch_two_class, boolean_one, boolean_two, branch_one_grads, branch_two_grads, epoch_count_one, epoch_count_two)
        branch_one_train_accuracies.append(branch_one_train_acc)
        branch_two_train_accuracies.append(branch_two_train_acc)
        branch_one_train_P.append(b1_train_P)
        branch_two_train_P.append(b2_train_P)
        branch_one_train_R.append(b1_train_R)
        branch_two_train_R.append(b2_train_R)
        branch_one_train_F.append(b1_train_F)
        branch_two_train_F.append(b2_train_F)
        optimizer, branch_one_val_acc, branch_two_val_acc, b1_test_P, b2_test_P, b1_test_R, b2_test_R, b1_test_F, b2_test_F, boolean_one, boolean_two, branch_one_grads, branch_two_grads, epoch_count_one, epoch_count_two = test_congestion_avoider(start_time, testloader, device, model, optimizer, scheduler, branch_one_grads, branch_two_grads, branch_one_class, branch_two_class, branch_one_criterion, branch_two_criterion, epoch, epochs, min_cond, max_cond, min_epochs, mult_factor, metric, epoch_count_one, epoch_count_two)
        branch_one_test_accuracies.append(branch_one_val_acc)
        branch_two_test_accuracies.append(branch_two_val_acc)
        branch_one_test_P.append(b1_test_P)
        branch_two_test_P.append(b2_test_P)
        branch_one_test_R.append(b1_test_R)
        branch_two_test_R.append(b2_test_R)
        branch_one_test_F.append(b1_test_F)
        branch_two_test_F.append(b2_test_F)

        branch_one_condition.append(boolean_one)
        branch_two_condition.append(boolean_two)

    return branch_one_train_accuracies, branch_two_train_accuracies, branch_one_train_P, branch_two_train_P, branch_one_train_R, branch_two_train_R, branch_one_train_F, branch_two_train_F, branch_one_test_accuracies, branch_two_test_accuracies, branch_one_test_P, branch_two_test_P, branch_one_test_R, branch_two_test_R, branch_one_test_F, branch_two_test_F, branch_one_condition, branch_two_condition


def get_cong_avoidance_results_10classes(epochs=100, min_cond=0.5, max_cond = 0.5, mult=0.1, lr=0.1, min_epochs = 10, num_class_avg = 10, similarity_threshold = 0.2, metric='recall'):

    '''
    A function to produce the results of training ResNet model 
    using the congestion avoidance scheduler for the long tailed CIFAR-10 data.

    Inputs:
        epochs: int - the number of epochs the model is trained for
        min_cond: float - the lower bound of the congestion condition parameter that will be interpolated between
        max_cond: float - the upper bound of the congestion condition parameter that will be interpolated between
        mult: float - the multiplicative decrease factor that determines the proportion of knowledge returned
        lr: float - The initial learning rate
        min_epochs: int - the minimum number of epochs that must pass between successive congestion events
        num_class_avg: int - the number of classes used to defined the congestion threshold value
        similarity_threshold: float - the threshold of cosine similarity
        metric: ['recall','precision','F-score'] - parameter to determine which metric is used to determine a congestion event
    
    Returns:
        train_acc: numpy array - The list of training accuracy in each epoch
        train_P: numpy array - The list of training precisions for each image class in each epoch
        train_R: numpy array - The list of training recalls for each image class in each epoch
        train_F: numpy array - The list of training F-scores for each image class in each epoch
        test_acc: numpy array - The list of accuracies on test data in each epoch
        test_P: numpy array - The list of precisions on test data for each image class in each epoch
        test_R: numpy array - The list of recalls on test data for each image class in each epoch
        test_F: numpy array - The list of F-scores on test data for each image class in each epoch
        cong_events: numpy array - indicator to show which class has a congestion event in each epoch
    '''
    # Create ResNet model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18(num_classes=10)
    model = model.to(device)
    if device == 'cuda':
        print('CUDA device used...')
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Import data
    trainset, trainloader, testset, testloader = create_sampled_CIFAR10_data()

    # Create variables
    cls_num = len(trainset.classes)
    grads = {}
    for cls in range(cls_num):
        grads[cls] = {}
    epoch_counts = [0]*cls_num
    train_acc = np.zeros((epochs, 1))
    train_P = np.zeros((epochs, cls_num))
    train_R = np.zeros((epochs, cls_num))
    train_F = np.zeros((epochs, cls_num))
    test_acc = np.zeros((epochs, 1))
    test_P = np.zeros((epochs, cls_num))
    test_R = np.zeros((epochs, cls_num))
    test_F = np.zeros((epochs, cls_num))
    cong_events = np.zeros((epochs, cls_num))
    
    criterion= nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=5e-4)
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=lr, step_size_up=10, mode="triangular2")

    # BEGIN RECORDING THE TIME
    start_time = time.time()

    for epoch in range(epochs):
        print('\n********** EPOCH {} **********'.format(epoch + 1))
        print('Learning rate: ', optimizer.param_groups[0]['lr'])
        print('Epochs since last congestion event: ', epoch_counts)
        confusion_matrix, accuracy, recalls, precisions, fScores, grads, epoch_counts = train_congestion_avoider_10_classes(device, model, trainloader, criterion, optimizer, cls_num, epoch_counts, grads)
        train_acc[epoch] = accuracy
        train_P[epoch] = precisions
        train_R[epoch] = recalls
        train_F[epoch] = fScores
        optimizer, accuracy, precisions, recalls, fScores, boolean_values, grads, epoch_counts = test_congestion_avoider_10classes(cls_num, start_time, testloader, device, model, optimizer, scheduler, grads, criterion, epoch, epochs, min_cond, max_cond, min_epochs, mult, epoch_counts, num_class_avg, similarity_threshold, metric)
        test_acc[epoch] = accuracy
        test_P[epoch] = precisions
        test_R[epoch] = recalls
        test_F[epoch] = fScores
        cong_events[epoch] = boolean_values

    return train_acc, train_P, train_R, train_F, test_acc, test_P, test_R, test_F, cong_events
