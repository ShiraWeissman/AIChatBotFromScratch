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



# Extract configurations





# Custom dataset class (loads preprocessed data directly)
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

def prepare_for_training(task_type):
    print('Loading training configuration..')
    with open(os.path.join(root_path, "config", "distilbert_training_config.yaml"), "r") as config_file:
        config = yaml.safe_load(config_file)
    if task_type == 'language_modeling':
        config = config['language_modeling']
    elif task_type == 'question_answering':
        config = config['question_answering']
    dataset_type = config["dataset_type"]
    dataset_name = config["dataset_name"]
    data_path = config["data_path"]
    checkpoint_dir = config["checkpoint_dir"]
    pretrained_model_name = config["pretrained_model_name"]
    model_type = config["model_type"]
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    batch_size = int(config["training"]["batch_size"])
    learning_rate = float(config["training"]["learning_rate"])
    # Ensure checkpoint directory exists
    os.makedirs(os.path.join(root_path, checkpoint_dir), exist_ok=True)
    print('Loading dataset..')
    dataset_path = os.path.join(root_path, data_path, f"{dataset_name}_{model_type}_preprocessed.pkl")
    dataset = CustomDataset(dataset_path, dataset_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print('Loading model..')
    model = load_model(task_type=dataset_type, pretrained_model_name=pretrained_model_name).to(device)

    print('Building optimizer..')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    return dataloader, model, optimizer, config


# Training Loop
def train(dataloader, model, optimizer, config):
    epochs = int(config["training"]["epochs"])
    device = config["device"]
    dataset_type = config["dataset_type"]
    dataset_type_short = 'lm' if dataset_type == "language_modeling" else 'qa'
    checkpoint_dir = config["checkpoint_dir"]
    dataset_name = config["dataset_name"]
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        # Wrap dataloader with tqdm for progress tracking
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if dataset_type == "language_modeling":
                labels = batch["labels"].to(device)
                loss, _ = model(input_ids, attention_mask, labels=labels)

            elif dataset_type == "question_answering":
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                loss, _, _ = model(input_ids, attention_mask, start_positions, end_positions)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(root_path, checkpoint_dir, f"{dataset_type}_{dataset_name}_epoch{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    model.save_pretrained(os.path.join(root_path, f"train/models/distilbert_{dataset_type_short}"))



if __name__ == "__main__":
    # task type options: "language_modeling" or "question_answering"
    dataloader, model, optimizer, config = prepare_for_training(task_type="question_answering")
    train(dataloader, model, optimizer, config)
