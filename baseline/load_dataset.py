import json


def load_dataset():
    train_filename = "../data/train_dataset.json"
    test_filename = "../data/test_dataset.json"
    
    # Load train dataset
    with open(train_filename, 'r') as file:
        train_dataset = json.load(file)
    
    # Load test dataset
    with open(test_filename, 'r') as file:
        test_dataset = json.load(file)
    
    return train_dataset, test_dataset
