import matplotlib.pyplot as plt
import pickle

def download_results(path, files, schedule_names):
    '''
    Results in each file are always in the following order:
    - Cat classifier training accuracy
    - Dog classifier training accuracy
    - Cat classifier test accuracy
    - Dog classifier test accuracy
    '''

    training = []
    testing = []
    labels = []

    for file, name in zip(files, schedule_names):
        labels.append('{} - (Cat)'.format(name))
        labels.append('{} - (Dog)'.format(name))
        with open(path + file, 'rb') as data:
            train_cat = pickle.load(data)
            training.append(train_cat)

            train_dog = pickle.load(data)
            training.append(train_dog)
            
            test_cat = pickle.load(data)
            testing.append(test_cat)

            test_dog = pickle.load(data)
            testing.append(test_dog)

    return training, testing, labels


def moving_average(data, window_size):
    
    index = 0
    moving_averages = []
    
    while index < len(data) - window_size + 1:
        window = data[index : index + window_size]
        average = sum(window) / window_size
        moving_averages.append(average)
        index += 1

    return moving_averages


def plot_accuracy_differences(files, schedule_names):
    '''
    Results in each file are always in the following order:
    - Cat classifier training accuracy
    - Dog classifier training accuracy
    - Cat classifier test accuracy
    - Dog classifier test accuracy
    '''

    path = '/Users/tom/Documents/Liverpool University/Dissertation/Code Notebooks/Accuracies - Split ResNet/5000 - 5000 -5000 dataset (shared layers)/SHARED__'

    baseline = [0] * 95

    plt.plot(baseline, 'k--')

    for file, name in zip(files, schedule_names):
        with open(path + file, 'rb') as data:
            train_cat = pickle.load(data)
            train_dog = pickle.load(data)
            test_cat = pickle.load(data)
            test_dog = pickle.load(data)
            difference = [dog - cat for cat, dog in zip(test_cat, test_dog)]
            plt.plot(moving_average(difference, 5), label=name)

    plt.xlabel('Epoch')
    plt.ylabel('Dog accuracy - Cat accuracy (%)')
    plt.xlim([0,100])
    plt.ylim([-10,10])
    plt.legend()
    plt.title('Difference in validation accuracies (dog vs cat classifier)')
    plt.show()


def plot_results(training, testing, labels):
    '''
    Results in each list are always in the following order:
    - Cat classifier training accuracy
    - Dog classifier training accuracy
    '''


    colors = ['black', 'red', 'darkorange', 'gold',
            'green', 'royalblue', 'purple', 'grey']

    #for index, train in enumerate(training):
    #    plt.plot(train, label='Training: ' + labels[index], color=colors[index])
    for index, test in enumerate(testing):
        plt.plot(test, label='Test: ' + labels[index], color=colors[index // 2])        
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc="upper left")    
    plt.show()


if __name__ == '__main__':
    plot_accuracy_differences(['constant_LR.pickle', 'expDecay_LR.pickle', 'mult_LR.pickle', 'multiStep_LR.pickle', 'plateau_LR.pickle', 'triangular_decayed_LR.pickle', 'cosineWarmRestarts_LR.pickle', 'RMSprop_LR.pickle', 'ADAM_LR.pickle'], 
                                                ['Constant', 'Exponential', 'Multiplicative', 'Multi-Step', 'Reduce on Plateau', 'Cyclic', 'Cosine Annealing', 'RMSprop', 'ADAM'])