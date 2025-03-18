import os
import json
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from models.distilbert_model.model import load_model
from transformers import DistilBertTokenizer
import torch.nn.functional as F
import argparse

# Load configuration from YAML
with open("training_config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract configurations
DATASET_TYPE = config["dataset_type"]
DATA_PATH = config["data_path"]
CHECKPOINT_DIR = config["checkpoint_dir"]
MODEL_NAME = config["model_name"]
DEVICE = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


# Custom dataset class for inference (handles batch inputs)
class InferenceDataset(Dataset):
    def __init__(self, input_data):
        self.data = input_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if DATASET_TYPE == "language_modeling":
            inputs = tokenizer(item['text'], return_tensors='pt', padding=True, truncation=True)
            return inputs

        elif DATASET_TYPE == "question_answering":
            inputs = tokenizer(item['question'], item['context'], return_tensors='pt', padding=True, truncation=True)
            return inputs


# Load input data for inference
def load_input_data(input_file):
    with open(input_file, 'r') as f:
        return json.load(f)


# Load model
def load_trained_model():
    latest_checkpoint = sorted(os.listdir(CHECKPOINT_DIR))[-1]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
    model = load_model(task_type=DATASET_TYPE, model_name=MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


# Inference function
def infer(model, dataloader):
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(DEVICE)  # Remove batch dimension
            attention_mask = batch['attention_mask'].squeeze(1).to(DEVICE)

            if DATASET_TYPE == "language_modeling":
                logits = model(input_ids, attention_mask=attention_mask).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                predictions.append(predicted_text)

            elif DATASET_TYPE == "question_answering":
                start_logits, end_logits = model(input_ids, attention_mask=attention_mask)
                start_pred = torch.argmax(start_logits, dim=-1)
                end_pred = torch.argmax(end_logits, dim=-1)

                for i in range(len(start_pred)):
                    answer = tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(input_ids[i][start_pred[i]: end_pred[i] + 1])
                    )
                    predictions.append(answer)
    return predictions


# Main function
def main(input_file):
    # Load input data for inference
    input_data = load_input_data(input_file)

    # Create dataset and dataloader
    dataset = InferenceDataset(input_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)  # Adjust batch size for inference

    # Load the trained model
    model = load_trained_model()

    # Get predictions
    predictions = infer(model, dataloader)

    # Print or save predictions
    for i, prediction in enumerate(predictions):
        print(f"Prediction {i + 1}: {prediction}")

    # Optionally save predictions to a file
    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)


# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on the trained model.")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input file for inference (JSON format)"
    )
    args = parser.parse_args()

    # Run the main function with the specified input file
    main(args.input_file)
