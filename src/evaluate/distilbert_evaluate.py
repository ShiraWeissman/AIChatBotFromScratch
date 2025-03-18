import os
import json
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from models.distilbert_model.model import load_model
from evaluate import load

# Load configuration from YAML
with open("training_config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract configurations
DATASET_TYPE = config["dataset_type"]
DATA_PATH = config["data_path"]
CHECKPOINT_DIR = config["checkpoint_dir"]
MODEL_NAME = config["model_name"]
DEVICE = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

BATCH_SIZE = config["training"]["batch_size"]

# Load evaluation metric
if DATASET_TYPE == "language_modeling":
    metric = load("perplexity", module_type="metric")  # Hugging Face `perplexity`
elif DATASET_TYPE == "question_answering":
    metric = load("squad_v2")  # SQuAD metrics for QA (Exact Match & F1-score)


# Custom dataset class (loads preprocessed test data)
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
                "context": item["context"],
                "question": item["question"],
                "answers": item["answers"],
            }


# Load test dataset
test_data_path = os.path.join(DATA_PATH, f"{DATASET_TYPE}_test.json")
test_dataset = CustomDataset(test_data_path)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load latest checkpoint
latest_checkpoint = sorted(os.listdir(CHECKPOINT_DIR))[-1]
checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
model = load_model(task_type=DATASET_TYPE, model_name=MODEL_NAME).to(DEVICE)
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()


# Evaluation function
def evaluate():
    total_loss = 0
    all_preds = []
    all_references = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            if DATASET_TYPE == "language_modeling":
                labels = batch["labels"].to(DEVICE)
                loss, logits = model(input_ids, attention_mask, labels=labels)
                total_loss += loss.item()

                # Compute Perplexity
                probs = F.softmax(logits, dim=-1)
                perplexity = torch.exp(-torch.sum(probs * torch.log(probs), dim=-1)).mean().item()
                metric.add_batch(predictions=perplexity, references=labels.cpu().numpy())

            elif DATASET_TYPE == "question_answering":
                start_positions = batch["start_positions"].to(DEVICE)
                end_positions = batch["end_positions"].to(DEVICE)
                loss, start_logits, end_logits = model(input_ids, attention_mask, start_positions, end_positions)
                total_loss += loss.item()

                # Convert logits to answer spans
                start_pred = torch.argmax(start_logits, dim=-1)
                end_pred = torch.argmax(end_logits, dim=-1)

                for i in range(len(batch["context"])):
                    context = batch["context"][i]
                    predicted_answer = context[start_pred[i]: end_pred[i] + 1]
                    all_preds.append(predicted_answer)
                    all_references.append(batch["answers"][i])

    # Compute final metric
    if DATASET_TYPE == "language_modeling":
        perplexity = metric.compute()
        print(f"Perplexity: {perplexity:.4f}")

    elif DATASET_TYPE == "question_answering":
        results = metric.compute(predictions=all_preds, references=all_references)
        print(f"Exact Match (EM): {results['exact_match']:.2f}")
        print(f"F1 Score: {results['f1']:.2f}")

    avg_loss = total_loss / len(test_dataloader)
    print(f"Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    evaluate()

