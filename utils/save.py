import pickle

def save_to_file(fileName, results):
    with open(fileName, 'wb') as file:
        for result in results:
            pickle.dump(result, file)