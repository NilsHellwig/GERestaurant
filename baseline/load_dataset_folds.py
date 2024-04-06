import json


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
    with open('../data/dataset_filtered.json', 'r') as json_datei:
        dataset = json.load(json_datei)

    # 2. Shuffle Dataset
    random.shuffle(dataset)

    # 3. Split dataset into folds
    fold_size = len(dataset) // 5
    folds = [dataset[i*fold_size:(i+1)*fold_size] for i in range(5)]

    train_datasets = []
    test_datasets = []

    for i in range(5):
        # Use ith fold as test set
        test_dataset = folds[i]

        # Use other folds for training
        train_dataset = []
        for j in range(5):
            if j != i:
                train_dataset.extend(folds[j])

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
        print(len(set([tag["tag_with_polarity"] for example in test_dataset for tag in example["tags"]])))


    check_for_duplicate_ids(train_datasets, test_datasets)

    return train_datasets, test_datasets
