import os
import pickle
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer
from models.distilbert_model.model import load_model
from tqdm import tqdm
from src.utils import *


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
    pretrained_model_name = config["pretrained_model_name"]
    pretrained_tokenizer_name = config["pretrained_tokenizer_name"]
    model_type = config["model_type"]
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    print('Loading dataset..')
    dataset_path = os.path.join(root_path, data_path, f"{dataset_name}_{model_type}_preprocessed")
    train_dataset = CustomDataset(dataset_path, dataset_type, split='train', model_type='distilbert')
    valid_dataset = CustomDataset(dataset_path, dataset_type, split='validation', model_type='distilbert')

    print('Loading model..')
    model = load_model(task_type=dataset_type, pretrained_model_name=pretrained_model_name,
                       pretrained_tokenizer_name=pretrained_tokenizer_name).to(device)

    return model, train_dataset, valid_dataset, config
# Training Loop
def train_model(model, train_dataset, valid_dataset, config):
    """
    Trains DistilGPT-2 using a pre-tokenized dataset.
    """
    save_path = os.path.join(root_path, config["save_path"])

    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(config['training']['learning_rate']),  # Recommended learning rate for fine-tuning
        weight_decay=float(config['training']['weight_decay']),  # Weight decay to prevent overfitting
        warmup_steps=int(config['training']['warmup_steps']),  # Gradual warm-up of learning rate
        lr_scheduler_type=config['training']['lr_scheduler_type'],  # Cosine learning rate decay
        per_device_train_batch_size=int(config['training']['per_device_train_batch_size']),
        per_device_eval_batch_size=int(config['training']['per_device_train_batch_size']),
        num_train_epochs=int(config["training"]["epochs"]),
        save_total_limit=int(config["training"]["save_total_limit"]),
        logging_dir=f"{save_path}/logs",
        logging_steps=int(config["training"]["logging_steps"]),
        report_to=config["report_to"] if bool(config["report_to"]) else "none",
        bf16=bool(config["training"]["bf16"]),
        optim=config["training"]["optim"]
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=model.tokenizer
    )

    trainer.train()
    model.save_model(os.path.join(root_path, save_path))
    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    # task type options: "language_modeling" or "question_answering"
    model, train_dataset, valid_dataset, config = prepare_for_training(task_type="question_answering")
    train_model(model, train_dataset, valid_dataset, config)
