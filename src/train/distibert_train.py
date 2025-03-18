import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.distilbert_model.model import load_model

# Load configuration from YAML
with open("distilbert_training_config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract configurations
DATASET_TYPE = config["dataset_type"]
DATA_PATH = config["data_path"]
CHECKPOINT_DIR = config["checkpoint_dir"]
MODEL_NAME = config["model_name"]
DEVICE = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LEARNING_RATE = config["training"]["learning_rate"]

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Custom dataset class (loads preprocessed data directly)
class CustomDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if DATASET_TYPE == "language_modeling":
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(item["labels"], dtype=torch.long),
            }

        elif DATASET_TYPE == "question_answering":
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "start_positions": torch.tensor(item["start_position"], dtype=torch.long),
                "end_positions": torch.tensor(item["end_position"], dtype=torch.long),
            }


# Load dataset
dataset_path = os.path.join(DATA_PATH, f"{DATASET_TYPE}.json")
dataset = CustomDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load model
model = load_model(task_type=DATASET_TYPE, model_name=MODEL_NAME).to(DEVICE)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# Training Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
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
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{DATASET_TYPE}_epoch{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    train()
