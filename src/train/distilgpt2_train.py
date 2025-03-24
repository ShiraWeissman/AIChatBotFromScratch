import yaml
from transformers import TrainingArguments, Trainer
from src.utils import *
from models.distilgpt2_model.model import load_model


def prepare_for_training(task_type):
    print('Loading training configuration..')
    with open(os.path.join(root_path, "config", "distilgpt2_training_config.yaml"), "r") as config_file:
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
    train_dataset = CustomDataset(dataset_path, dataset_type, split='train')
    valid_dataset = CustomDataset(dataset_path, dataset_type, split='validation')

    print('Loading model..')
    model = load_model(task_type=dataset_type, pretrained_model_name=pretrained_model_name).to(device)

    return model, train_dataset, valid_dataset, config


def train_model(model, train_dataset, valid_dataset, config):
    """
    Trains DistilGPT-2 using a pre-tokenized dataset.
    """
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    device = config["device"]
    dataset_type = config["dataset_type"]
    dataset_type_short = 'lm' if dataset_type == "language_modeling" else 'qa'
    checkpoint_dir = config["checkpoint_dir"]
    dataset_name = config["dataset_name"]
    save_path = os.path.join(root_path, config["save_path"])
    # # Load preprocessed dataset
    # tokenized_dataset = dataloader
    #
    # # Set dataset format for PyTorch
    # tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_total_limit=2,
        logging_dir=f"{save_path}/logs",
        logging_steps=100
    )

    # Trainer setup
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=model.tokenizer
    )

    # Train and save
    trainer.train()
    model.save_model(os.path.join(root_path, save_path))
    print(f"Model saved at {save_path}")


if __name__ == '__main__':
    model, train_dataset, valid_dataset, config = prepare_for_training(task_type="language_modeling")
    train_model(model, train_dataset, valid_dataset, config)
