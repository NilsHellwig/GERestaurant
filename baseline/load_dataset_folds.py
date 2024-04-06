import constants
import json

# This function double-checks if there are any training examples that are also used for testing
# - this should for sure not be the case, however I'm checking it here once more


def check_for_duplicate_ids(train_dataset, test_dataset):
    for split_id in range(len(train_dataset)):
        ids_train = [example["id"] for example in train_dataset[split_id]]
        ids_test = [example["id"] for example in test_dataset[split_id]]

        # Check for duplicates within each dataset
        if len(ids_train) != len(set(ids_train)):
            raise ValueError("Duplicates found in training data.")

        if len(ids_test) != len(set(ids_test)):
            raise ValueError("Duplicates found in test data.")

        # Check for common ids across datasets
        common_ids = set(ids_train) & set(ids_test)

        if len(common_ids):
            raise ValueError(
                "Ids used in both training and test data: " + str(common_ids))


def load_dataset_folds(random):
    # 1. Load Dataset
    with open(f'../data/dataset_filtered.json', 'r') as json_datei:
        dataset = json.load(json_datei)
    
    # 2. Create Splits
    print(len(dataset))
    raise KeyError

    print("Train:", len(train_dataset[0]), len(train_dataset))
    print("Test:", len(test_dataset[0]), len(test_dataset))

    check_for_duplicate_ids(train_dataset, test_dataset)

    return train_dataset, test_dataset
