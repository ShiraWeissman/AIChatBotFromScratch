import os
import torch
import pickle
from torch.utils.data import DataLoader, Dataset

root_path = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))


class CustomDataset(Dataset):
    def __init__(self, data_path, dataset_type, split="train"):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        if split not in ["train", "validation"]:
            raise ValueError("split must be either 'train' or 'validation'")

        self.dataset_type = dataset_type
        self.split = split
        self.dataset = self.data[self.split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.dataset_type == "language_modeling":
            print()
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(item["input_ids"], dtype=torch.long),
            }

        elif self.dataset_type == "question_answering":
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "start_positions": torch.tensor(item["start_positions"], dtype=torch.long),
                "end_positions": torch.tensor(item["end_positions"], dtype=torch.long),
            }

        else:
            raise ValueError("Unsupported dataset_type")


if __name__ == '__main__':
    pass