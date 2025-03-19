import os
import pickle
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.distilbert_model.model import load_model
from tqdm import tqdm
from src.utils import *

# Load configuration from YAML
with open(os.path.join(ROOT_PATH, "config", "distilbert_training_config.yaml"), "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract configurations
DATASET_TYPE = config["dataset_type"]
DATASET_NAME = config["dataset_name"]
DATA_PATH = config["data_path"]
CHECKPOINT_DIR = config["checkpoint_dir"]
MODEL_NAME = config["model_name"]
MODEL_TYPE = config["model_type"]
DEVICE = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

BATCH_SIZE = int(config["training"]["batch_size"])
EPOCHS = int(config["training"]["epochs"])
LEARNING_RATE = float(config["training"]["learning_rate"])

# Ensure checkpoint directory exists
os.makedirs(os.path.join(ROOT_PATH, CHECKPOINT_DIR), exist_ok=True)


# Custom dataset class (loads preprocessed data directly)
class CustomDataset(Dataset):
    def __init__(self, data_path, split="train"):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        if split not in ["train", "validation"]:
            raise ValueError("split must be either 'train' or 'validation'")

        self.split = split
        self.dataset = self.data[self.split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if DATASET_TYPE == "language_modeling":
            print()
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(item["input_ids"], dtype=torch.long),
            }

        elif DATASET_TYPE == "question_answering":
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "start_positions": torch.tensor(item["start_position"], dtype=torch.long),
                "end_positions": torch.tensor(item["end_position"], dtype=torch.long),
            }

        else:
            raise ValueError("Unsupported DATASET_TYPE")

def prepare_for_training():
    print('Loading dataset..')
    dataset_path = os.path.join(ROOT_PATH, DATA_PATH, f"{DATASET_NAME}_{MODEL_TYPE}_preprocessed.pkl")
    dataset = CustomDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print('Loading model..')
    model = load_model(task_type=DATASET_TYPE, model_name=MODEL_NAME).to(DEVICE)

    print('Building optomizer..')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    return dataloader, model, optimizer


# Training Loop
def train(dataloader, model, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0

        # Wrap dataloader with tqdm for progress tracking
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            if DATASET_TYPE == "language_modeling":
                labels = batch["labels"].to(DEVICE)
                loss, _ = model(input_ids, attention_mask, labels=labels)

            elif DATASET_TYPE == "question_answering":
                start_positions = batch["start_positions"].to(DEVICE)
                end_positions = batch["end_positions"].to(DEVICE)
                loss, _, _ = model(input_ids, attention_mask, start_positions, end_positions)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(ROOT_PATH, CHECKPOINT_DIR, f"{DATASET_TYPE}_{DATASET_NAME}_epoch{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    dataloader, model, optimizer = prepare_for_training()
    train(dataloader, model, optimizer)
