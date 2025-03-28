import os
import torch
import pickle
import yaml
import shutil
from torch.utils.data import Dataset

root_path = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))


class CustomDataset(Dataset):
    def __init__(self, data_path, dataset_type, split="train", model_type='distilgpt2'):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        if split not in ["train", "validation"]:
            raise ValueError("split must be either 'train' or 'validation'")

        self.dataset_type = dataset_type.lower()
        self.split = split
        self.dataset = self.data[self.split]
        self.model_type = model_type

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
            if self.model_type == 'distilbert':
                return {
                    "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                    "start_positions": torch.tensor(item["start_positions"], dtype=torch.long),
                    "end_positions": torch.tensor(item["end_positions"], dtype=torch.long),
                }
            elif self.model_type == 'distilgpt2':
                return {
                    "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(item["input_ids"], dtype=torch.long),
                }

        else:
            raise ValueError("Unsupported dataset_type")

def load_config(config_name):
    with open(os.path.join(root_path, "config", f"{config_name}.yaml"), "r") as f:
        config = yaml.safe_load(f)
    return config

def zip_folder(folder_to_zip):
    base_dir = os.path.dirname(folder_to_zip)
    folder_name = os.path.basename(folder_to_zip)
    shutil.make_archive(folder_to_zip, 'zip',  folder_name)

def extract_zipped_folder(zipped_folder_path):
    shutil.unpack_archive(zipped_folder_path, zipped_folder_path.split('.')[0])

if __name__ == '__main__':
    pass