import pickle

def save_to_file(fileName, results):
    '''
    A function to save the results of the model training to a pickle file

    Inputs:
        fileName: string - the file that the data will be saved to
        results: list - the list of data items to be saved to the file
    
    Returns:
        A pickle file of the specified name is created and saved
    '''
    
    with open(fileName, 'wb') as file:
        for result in results:
            pickle.dump(result, file)