import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle
import numpy as np


def moving_average(data, smooth_param):

    '''
    A function to calculate the moving average of a list of data

    Inputs:
      data: the raw data that will be smoothed
      smooth_param: int - the number of data items used to create each moving average value

    Returns:
      moving_averages: list - the smoothed data
    '''
    
    index = 0
    moving_averages = []
    
    while index < len(data) - smooth_param + 1:
        sub_data = data[index : index + smooth_param]
        average = sum(sub_data) / smooth_param
        moving_averages.append(average)
        index += 1

    return moving_averages


def plot_results_twoClass(names, params, class_names, colors, smooth = 5):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,6))

    for index, (name, param) in enumerate(zip(names, params)):
        with open(name, 'rb') as data:
            train_acc_one = pickle.load(data)
            train_acc_two = pickle.load(data)
            train_P_one = pickle.load(data)
            train_P_two = pickle.load(data)
            train_R_one = pickle.load(data)
            train_R_two = pickle.load(data)
            train_F_one = pickle.load(data)
            train_F_two = pickle.load(data)
            test_acc_one = pickle.load(data)
            test_acc_two = pickle.load(data)
            test_P_one = pickle.load(data)
            test_P_two = pickle.load(data)
            test_R_one = pickle.load(data)
            test_R_two = pickle.load(data)
            test_F_one = pickle.load(data)
            test_F_two = pickle.load(data)
            congestion_one = pickle.load(data)
            congestion_two = pickle.load(data)

        ax[0,0].plot(moving_average(test_acc_one,smooth), color = colors[index], label='{}: {}'%(class_names[0],format(param)))
        ax[0,0].plot(moving_average(test_acc_two,smooth), color = colors[index], linestyle='--', label='{}: {}'%(class_names[1],format(param)))
        ax[0,0].set_title('Accuracies \n(moving average of over {} epochs)'.format(smooth))
        ax[0,1].plot(moving_average(test_P_one,smooth), colors[index], label='{}: {}'%(class_names[0],format(param)))
        ax[0,1].plot(moving_average(test_P_two,smooth), colors[index], linestyle='--', label='{}: {}'%(class_names[1],format(param)))
        ax[0,1].set_title('Precision \n(moving average of over {} epochs)'.format(smooth))
        ax[1,0].plot(moving_average(test_R_one,smooth), colors[index], label='{}: {}'%(class_names[0],format(param)))
        ax[1,0].plot(moving_average(test_R_two,smooth), colors[index], linestyle='--', label='{}: {}'%(class_names[1],format(param)))
        ax[1,0].set_title('Recall \n(moving average of over {} epochs)'.format(smooth))
        ax[1,1].plot(moving_average(test_F_one,smooth), colors[index], label='{}: {}'%(class_names[0],format(param)))
        ax[1,1].plot(moving_average(test_F_two,smooth), colors[index], linestyle='--', label='{}: {}'%(class_names[1],format(param)))
        ax[1,1].set_title('F-Score \n(moving average of over {} epochs)'.format(smooth))

        if len(names) == 2:
            for i, val in enumerate(congestion_one):
                if val == 1:
                    ax[0,0].axvline(x=i, color=colors[index], linewidth=0.5)
                    ax[0,1].axvline(x=i, color=colors[index], linewidth=0.5)
                    ax[1,0].axvline(x=i, color=colors[index], linewidth=0.5)
                    ax[1,1].axvline(x=i, color=colors[index], linewidth=0.5)
            for i, val in enumerate(congestion_two):
                if val == 1:
                    ax[0,0].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)
                    ax[0,1].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)
                    ax[1,0].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)
                    ax[1,1].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)

        fig.text(0.4, 0.04, 'Epoch', ha='center', va='center')
        fig.text(0.06, 0.5, '%', ha='center', va='center', rotation='vertical')
        
        #ax[0,0].set_ylim([60,100])
        #ax[0,1].set_ylim([50,90])
        #ax[1,0].set_ylim([70,100])
        #ax[1,1].set_ylim([60,90])

        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        fig.tight_layout()

    plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
    plt.show()


def plot_results_diff_twoClass(names, params, colors, smooth = 5):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,6))

    for index, (name, param) in enumerate(zip(names, params)):
        with open(name, 'rb') as data:
            train_acc_one = pickle.load(data)
            train_acc_two = pickle.load(data)
            train_P_one = pickle.load(data)
            train_P_two = pickle.load(data)
            train_R_one = pickle.load(data)
            train_R_two = pickle.load(data)
            train_F_one = pickle.load(data)
            train_F_two = pickle.load(data)
            test_acc_one = pickle.load(data)
            test_acc_two = pickle.load(data)
            test_P_one = pickle.load(data)
            test_P_two = pickle.load(data)
            test_R_one = pickle.load(data)
            test_R_two = pickle.load(data)
            test_F_one = pickle.load(data)
            test_F_two = pickle.load(data)
            congestion_one = pickle.load(data)
            congestion_two = pickle.load(data)

        acc_diff = [dog - cat for cat, dog in zip(test_acc_one, test_acc_two)]
        precision_diff = [dog - cat for cat, dog in zip(test_P_one, test_P_two)]
        recall_diff = [dog - cat for cat, dog in zip(test_R_one, test_R_two)]
        fScore_diff = [dog - cat for cat, dog in zip(test_F_one, test_F_two)]

        ax[0,0].plot(moving_average(acc_diff,smooth), color = colors[index], label='{}'.format(param))
        ax[0,0].plot([0]*len(moving_average(acc_diff,smooth)), color='grey', linestyle='--')
        ax[0,0].set_title('Difference in accuracies \n(moving average of over {} epochs)'.format(smooth))

        ax[0,1].plot(moving_average(precision_diff,smooth), colors[index], label='{}'.format(param))
        ax[0,1].plot([0]*len(moving_average(precision_diff,smooth)), color='k', linestyle='--')
        ax[0,1].set_title('Difference in Precision \n(moving average of over {} epochs)'.format(smooth))
        
        ax[1,0].plot(moving_average(recall_diff,smooth), colors[index], label='{}'.format(param))
        ax[1,0].plot([0]*len(moving_average(recall_diff,smooth)), color='grey', linestyle='--')
        ax[1,0].set_title('Difference in Recall \n(moving average of over {} epochs)'.format(smooth))
        
        ax[1,1].plot(moving_average(fScore_diff,smooth), colors[index], label='{}'.format(param))
        ax[1,1].plot([0]*len(moving_average(fScore_diff,smooth)), color='grey', linestyle='--')
        ax[1,1].set_title('Difference in F-Score \n(moving average of over {} epochs)'.format(smooth))
        
        if len(names) == 2:
            for i, val in enumerate(congestion_one):
                if val == 1:
                    ax[0,0].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)
                    ax[0,1].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)
                    ax[1,0].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)
                    ax[1,1].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)
            for i, val in enumerate(congestion_two):
                if val == 1:
                    ax[0,0].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)
                    ax[0,1].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)
                    ax[1,0].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)
                    ax[1,1].axvline(x=i, color=colors[index], linestyle='--', linewidth=0.5)


        fig.text(0.4, 0.04, 'Epoch', ha='center', va='center')
        fig.text(0.06, 0.5, '%', ha='center', va='center', rotation='vertical')

        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        fig.tight_layout()

    #ax[0,0].set_ylim([-5,5])
    #ax[0,1].set_ylim([-10,10])
    #ax[1,1].set_ylim([-2,30])
    #ax[1,0].set_ylim([-5,50])

    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
    plt.show()


def plot_results_multiClass(name, colors, smooth = 5, vlines=True):

    '''
    A function to plot the performance metrics of the trained model for each class of images on the test data

    Inputs:
      name: string - the name of the file that contains the data to create the plots
      colors: list - A list of colors used for each line on the plots
      smooth: int - the level of smoothing used in the moving average function to smooth the data
      vlines: boolean - indicator to determine whether vertical lines are shown on the plot to show which epochs congestion events occur in

    Returns:
      None
    '''

    # Create a 2x2 subplot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,6))
    idx_to_class = {0: 'airplane', 1: 'automobile', 
                    2: 'bird', 3: 'cat', 
                    4: 'deer', 5: 'dog', 
                    6: 'frog', 7: 'horse', 
                    8: 'ship', 9: 'truck'}

    # Read the pickle file to create the required variables
    with open(name, 'rb') as data:
        train_acc = pickle.load(data)
        train_P = pickle.load(data)
        train_R = pickle.load(data)
        train_F = pickle.load(data)
        test_acc = pickle.load(data)
        test_P = pickle.load(data)
        test_R = pickle.load(data)
        test_F = pickle.load(data)
        cong_events = pickle.load(data)

    cls_num = int(train_P[0].shape[0])
    
    # Plot the accuracy of the model
    ax[0,0].plot(moving_average(test_acc,smooth), color='k')
    ax[0,0].set_title('Accuracy \n(moving average of over {} epochs)'.format(smooth))
    ax[0,0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # For each class of images plot the smoothed precision, recall & F-score in the corresponding subplot
    for cls in range(cls_num):
        ax[0,1].plot(moving_average(test_P[:,cls],smooth), color = colors[cls], label='Class {}: {}'.format(cls,idx_to_class[cls]))
        ax[1,0].plot(moving_average(test_R[:,cls],smooth), color = colors[cls], label='Class {}: {}'.format(cls,idx_to_class[cls]))
        ax[1,1].plot(moving_average(test_F[:,cls],smooth), color = colors[cls], label='Class {}: {}'.format(cls,idx_to_class[cls]))
    
    # If a congestion event has happened in that epoch then a vertical line is added to the graph at that epoch
    cong_events = np.sum(cong_events,1)
    if vlines:
        for i, event in enumerate(cong_events):
            if event > 0:
                ax[0,0].axvline(x=i, color='grey', linestyle='--', linewidth=0.5)
                ax[0,1].axvline(x=i, color='grey', linestyle='--', linewidth=0.5)
                ax[1,0].axvline(x=i, color='grey', linestyle='--', linewidth=0.5)
                ax[1,1].axvline(x=i, color='grey', linestyle='--', linewidth=0.5)
        
    # Add titles, axis formatting and legends to the plots
    ax[0,1].set_title('Precision \n(moving average of over {} epochs)'.format(smooth))
    ax[1,0].set_title('Recall \n(moving average of over {} epochs)'.format(smooth))
    ax[1,1].set_title('F-score \n(moving average of over {} epochs)'.format(smooth))
    ax[0,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax[1,0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax[1,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    fig.text(0.5, 0.04, 'Epoch', ha='center', va='center')
    handles, labels = ax[1,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    fig.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    plt.show()


def plot_results_diff_multiClass(baseline, name, colors, smooth = 5, vlines=True):

    '''
    A function to plot the difference in performance metrics for the trained model using
    the congestion avoidance scheduler with the model without the congestion avoidance scheduler.

    Inputs:
      baseline: string - the name of the file that contains the data for the baseline position (i.e. no congestion avoidance)
      name: string - the name of the file that contains the data to create the plots
      colors: list - A list of colors used for each line on the plots
      smooth: int - the level of smoothing used in the moving average function to smooth the data
      vlines: boolean - indicator to determine whether vertical lines are shown on the plot to show which epochs congestion events occur in

    Returns:
      None
    '''

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,6))
    idx_to_class = {0: 'airplane', 1: 'automobile', 
                    2: 'bird', 3: 'cat', 
                    4: 'deer', 5: 'dog', 
                    6: 'frog', 7: 'horse', 
                    8: 'ship', 9: 'truck'}

    # Read the pickle file containing the results when no congestion avoidance scheduler is used
    with open(baseline, 'rb') as baseline:
        base_train_acc = pickle.load(baseline)
        base_train_P = pickle.load(baseline)
        base_train_R = pickle.load(baseline)
        base_train_F = pickle.load(baseline)
        base_test_acc = pickle.load(baseline)
        base_test_P = pickle.load(baseline)
        base_test_R = pickle.load(baseline)
        base_test_F = pickle.load(baseline)
    
    # Read the pickle file containing the results when the congestion avoidance scheduler is used
    with open(name, 'rb') as data:
        train_acc = pickle.load(data)
        train_P = pickle.load(data)
        train_R = pickle.load(data)
        train_F = pickle.load(data)
        test_acc = pickle.load(data)
        test_P = pickle.load(data)
        test_R = pickle.load(data)
        test_F = pickle.load(data)
        cong_events = pickle.load(data)

    cls_num = int(train_P[0].shape[0])

    # Calculate the change to the accuracy in each epoch with the inclusion of congestion avoidance
    test_acc_diff = test_acc - base_test_acc
    
    ax[0,0].plot(moving_average(test_acc_diff,smooth), color='k')
    ax[0,0].set_title('Change to accuracy from baseline \n(moving average of over {} epochs)'.format(smooth))
    ax[0,0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # For each class plot the change in recall, precision and F-score
    for cls in range(cls_num):
        test_P_diff = test_P[:,cls] - base_test_P[:,cls]
        test_R_diff = test_R[:,cls] - base_test_R[:,cls]
        test_F_diff = test_F[:,cls] - base_test_F[:,cls]
        ax[0,1].plot(moving_average(test_P_diff,smooth), color = colors[cls], label='Class {}: {}'.format(cls,idx_to_class[cls]))
        ax[1,0].plot(moving_average(test_R_diff,smooth), color = colors[cls], label='Class {}: {}'.format(cls,idx_to_class[cls]))
        ax[1,1].plot(moving_average(test_F_diff,smooth), color = colors[cls], label='Class {}: {}'.format(cls,idx_to_class[cls]))
    
    # Add a vertical line to the subplots in each epoch that a congestion event occurs
    cong_events = np.sum(cong_events,1)
    if vlines:
        for i, event in enumerate(cong_events):
            if event > 0:
                ax[0,0].axvline(x=i, color='grey', linestyle='--', linewidth=0.5)
                ax[0,1].axvline(x=i, color='grey', linestyle='--', linewidth=0.5)
                ax[1,0].axvline(x=i, color='grey', linestyle='--', linewidth=0.5)
                ax[1,1].axvline(x=i, color='grey', linestyle='--', linewidth=0.5)
    
    # Add a dashed line along 0% to show the base position of no change in the metric values
    ax[0,0].plot([0]*len(moving_average(test_acc_diff,smooth)), color='grey', linestyle='--', linewidth=0.5)
    ax[0,1].plot([0]*len(moving_average(test_acc_diff,smooth)), color='grey', linestyle='--', linewidth=0.5)
    ax[1,0].plot([0]*len(moving_average(test_acc_diff,smooth)), color='grey', linestyle='--', linewidth=0.5)
    ax[1,1].plot([0]*len(moving_average(test_acc_diff,smooth)), color='grey', linestyle='--', linewidth=0.5)

    # Add titles, legends, axis formatting to the subplots
    ax[0,1].set_title('Change to precision from baseline \n(moving average of over {} epochs)'.format(smooth))
    ax[1,0].set_title('Change to recall from baseline \n(moving average of over {} epochs)'.format(smooth))
    ax[1,1].set_title('Change to F-score from baseline \n(moving average of over {} epochs)'.format(smooth))
    ax[0,1].set_ylim([-0.1,0.1])
    ax[1,0].set_ylim([-0.1,0.1])
    ax[1,1].set_ylim([-0.1,0.1])
    ax[0,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax[1,0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax[1,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    fig.text(0.5, 0.04, 'Epoch', ha='center', va='center')
    handles, labels = ax[1,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    fig.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    plt.show()
