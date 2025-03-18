import os
import pickle
from datasets import load_dataset
from transformers import DistilBertTokenizer
import numpy as np
from utils import *

# Define dataset names
DATASETS = {
    "tinystories": "roneneldan/TinyStories",
    "fairytaleqa": "FairyTaleQA/FairyTaleQA_1.0"
}

# Define paths for processed data storage
PROCESSED_DIR = os.path.join(ROOT_PATH, "data/processed/")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Initialize tokenizers (for DistilBERT & LSTM approaches)
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# Dummy function for LSTM (Word2Vec/GloVe embeddings would be handled separately)
def lstm_tokenize(text):
    return text.lower().split()


# Function to check if processed data exists
def check_existing_files(model_type):
    file_path = os.path.join(PROCESSED_DIR, f"{model_type}_preprocessed.pkl")
    return file_path if os.path.exists(file_path) else None


# Function to preprocess dataset
def preprocess_text(example, model_type="distilbert"):
    """Tokenizes text based on the chosen model type (DistilBERT or LSTM)."""
    if model_type == "distilbert":
        return distilbert_tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    elif model_type == "lstm":
        return {"tokens": lstm_tokenize(example["text"])}  # LSTM uses raw tokens
    return example


# Function to load and preprocess dataset
def load_and_preprocess_dataset(dataset_name, model_type="distilbert", force_reprocess=False):
    """
    Loads the dataset from Hugging Face and preprocesses it for the specified model type.
    :param dataset_name: Name of the dataset (e.g., 'tinystories' or 'fairytaleqa').
    :param model_type: 'distilbert' or 'lstm'.
    :param force_reprocess: If True, forces reprocessing even if a saved file exists.
    :return: Preprocessed dataset.
    """
    # Check for existing preprocessed file
    processed_file = check_existing_files(model_type)

    if processed_file and not force_reprocess:
        print(f"✅ Loading preprocessed {dataset_name} dataset from {processed_file}...")
        with open(processed_file, "rb") as f:
            return pickle.load(f)

    # Load dataset from Hugging Face
    dataset_path = DATASETS.get(dataset_name)
    if not dataset_path:
        raise ValueError(f"Dataset {dataset_name} not found in available datasets.")

    print(f"📥 Downloading {dataset_name} dataset from Hugging Face...")
    dataset = load_dataset(dataset_path)

    # Apply preprocessing
    print(f"🔄 Preprocessing {dataset_name} for {model_type} model...")
    dataset = dataset.map(lambda x: preprocess_text(x, model_type=model_type))

    # Save processed data
    save_path = os.path.join(PROCESSED_DIR, f"{model_type}_preprocessed.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"✅ Processed dataset saved at {save_path}")
    return dataset


# Example Usage
if __name__ == "__main__":
    # Load & preprocess for DistilBERT
    tinystories_distilbert = load_and_preprocess_dataset("tinystories", model_type="distilbert")
    fairytaleqa_distilbert = load_and_preprocess_dataset("fairytaleqa", model_type="distilbert")

    # Load & preprocess for LSTM
    tinystories_lstm = load_and_preprocess_dataset("tinystories", model_type="lstm")
    fairytaleqa_lstm = load_and_preprocess_dataset("fairytaleqa", model_type="lstm")
