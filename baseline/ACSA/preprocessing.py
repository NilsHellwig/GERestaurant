from torch.utils.data import Dataset as TorchDataset
import constants
import torch


class CustomDatasetACSA(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.encodings.items()}
        item["label"] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)


def aspect_category_sentiment_labels_to_one_hot(labels):
    one_hot = []

    for label in constants.ASPECT_CATEGORY_POLARITIES:
        if label in labels:
            one_hot.append(1)
        else:
            one_hot.append(0)
    return one_hot


def preprocess_data_ACSA(dataset, tokenizer):
    texts = [example["text"] for example in dataset]
    labels = [list(set([tag["tag_with_polarity"] for tag in example["tags"]]))
              for example in dataset]

    labels = [aspect_category_sentiment_labels_to_one_hot(
        label) for label in labels]
    labels = torch.tensor(labels, dtype=torch.float32)
    encodings = tokenizer(texts, padding=True, truncation=True,
                          max_length=constants.MAX_TOKENS_ACSA, return_tensors="pt")
    return CustomDatasetACSA(encodings, labels)
