import os
import pickle
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, AutoTokenizer
import numpy as np
from src.utils import *

# Define dataset names
DATASETS = {
    "tinystories": "roneneldan/TinyStories",
    "fairytaleqa": "GEM/FairytaleQA"
}

# Define paths for processed data storage
PROCESSED_DIR = os.path.join(root_path, "data/processed/")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Initialize tokenizers
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

distilgpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
distilgpt2_tokenizer.pad_token = distilgpt2_tokenizer.eos_token

# Dummy function for LSTM (Word2Vec/GloVe embeddings would be handled separately)
def lstm_tokenize(text):
    return text.lower().split()


# Function to check if processed data exists
def check_existing_files(model_type, dataset_name):
    file_path = os.path.join(PROCESSED_DIR, f"{dataset_name}_{model_type}_preprocessed.pkl")
    return file_path if os.path.exists(file_path) else None


# Function to preprocess dataset
def preprocess_text(raw, model_type="distilbert", dataset_name="tinystories"):
    """Preprocesses text for the chosen model type (DistilBERT for QA or LSTM) and dataset (TinyStories or FairytaleQA)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "distilbert":
        if dataset_name == "tinystories":
            # Tokenization for TinyStories using DistilBERT
            tokenized_output = distilbert_tokenizer(raw["text"], padding="max_length", truncation=True, max_length=512)

            # Move the tokenized output to the GPU (if available)
            tokenized_output = {key: torch.tensor(val).to(device) for key, val in tokenized_output.items()}
            return tokenized_output

        elif dataset_name == "fairytaleqa":
            # FairytaleQA-specific preprocessing for DistilBERT for Question Answering
            content = raw["content"]
            question = raw["question"]
            answer = raw["answer"]

            # Combine the content and question into a single string for tokenization
            input_text = f"content: {content} question: {question}"

            # Tokenize the combined text (context + question) using DistilBERT tokenizer
            tokenized_output = distilbert_tokenizer(input_text, padding="max_length", truncation=True, max_length=512)

            # Identify start and end positions for the answer in the tokenized context
            # Find the answer span in the tokenized input text (this is done by finding the start and end token indices)
            answer_tokens = distilbert_tokenizer(answer)["input_ids"]
            start_position = None
            end_position = None

            # Try to match the answer in the tokenized context to find start and end positions
            context_tokens = tokenized_output["input_ids"]
            answer_len = len(answer_tokens)

            for i in range(len(context_tokens) - answer_len + 1):
                if context_tokens[i:i + answer_len] == answer_tokens:
                    start_position = i
                    end_position = i + answer_len - 1
                    break

            # If no match is found, fallback to a default value (this can be adjusted based on your approach)
            if start_position is None or end_position is None:
                start_position = 0
                end_position = 0

            # Add start and end positions to the tokenized output
            tokenized_output["start_positions"] = torch.tensor(start_position).to(device)
            tokenized_output["end_positions"] = torch.tensor(end_position).to(device)

            # Move the tokenized output to the GPU (if available)
            tokenized_output = {key: torch.tensor(val).to(device) for key, val in tokenized_output.items()}
            return tokenized_output

    elif model_type == "distilgpt2":
        if dataset_name == "tinystories":
            # Tokenization for TinyStories using DistilGPT2
            tokenized_output = distilgpt2_tokenizer(raw["text"], truncation=True, padding="max_length", max_length=256)

            # Move the tokenized output to the GPU (if available)
            tokenized_output = {key: torch.tensor(val).to(device) for key, val in tokenized_output.items()}
            return tokenized_output
        elif dataset_name == "fairytaleqa":
            # FairytaleQA-specific preprocessing for DistilBERT for Question Answering
            content = raw["content"]
            question = raw["question"]
            answer = raw["answer"]

            # Combine the content and question into a single string for tokenization
            input_text = f"content: {content} question: {question}"

            # Tokenize the combined text (context + question) using DistilBERT tokenizer
            tokenized_output = distilgpt2_tokenizer(input_text,  truncation=True, padding="max_length", max_length=512)

            # Identify start and end positions for the answer in the tokenized context
            # Find the answer span in the tokenized input text (this is done by finding the start and end token indices)
            answer_tokens = distilbert_tokenizer(answer)["input_ids"]
            start_position = None
            end_position = None

            # Try to match the answer in the tokenized context to find start and end positions
            context_tokens = tokenized_output["input_ids"]
            answer_len = len(answer_tokens)

            for i in range(len(context_tokens) - answer_len + 1):
                if context_tokens[i:i + answer_len] == answer_tokens:
                    start_position = i
                    end_position = i + answer_len - 1
                    break

            # If no match is found, fallback to a default value (this can be adjusted based on your approach)
            if start_position is None or end_position is None:
                start_position = 0
                end_position = 0

            # Add start and end positions to the tokenized output
            tokenized_output["start_positions"] = torch.tensor(start_position).to(device)
            tokenized_output["end_positions"] = torch.tensor(end_position).to(device)

            # Move the tokenized output to the GPU (if available)
            tokenized_output = {key: torch.tensor(val).to(device) for key, val in tokenized_output.items()}
            return tokenized_output


    elif model_type == "lstm":
        if dataset_name == "tinystories":
            # Tokenization for TinyStories using LSTM
            tokens = lstm_tokenize(raw["text"])
            tokens_tensor = torch.tensor(tokens).to(device)
            return {"tokens": tokens_tensor}

        elif dataset_name == "fairytaleQA":
            # FairytaleQA-specific preprocessing for LSTM
            # Tokenize context (story) and question separately
            context = raw["story"]
            question = raw["question"]

            context_tokens = lstm_tokenize(context)
            question_tokens = lstm_tokenize(question)

            # Combine the tokens into a single tensor or process as needed
            context_tensor = torch.tensor(context_tokens).to(device)
            question_tensor = torch.tensor(question_tokens).to(device)

            return {"context_tokens": context_tensor, "question_tokens": question_tensor}

    return raw



# Function to load and preprocess dataset
def load_and_preprocess_dataset(dataset_name, model_type="distilgpt2", sample_size=10000, force_reprocess=False):
    """
    Loads the dataset from Hugging Face and preprocesses it for the specified model type.
    :param dataset_name: Name of the dataset (e.g., 'tinystories' or 'fairytaleqa').
    :param model_type: 'distilbert' or 'lstm'.
    :param force_reprocess: If True, forces reprocessing even if a saved file exists.
    :return: Preprocessed dataset.
    """
    # Check for existing preprocessed file
    processed_file = check_existing_files(model_type, dataset_name)

    if processed_file and not force_reprocess:
        print(f"Loading preprocessed {dataset_name} dataset from {processed_file}...")
        with open(processed_file, "rb") as f:
            return pickle.load(f)

    # Load dataset from Hugging Face
    dataset_path = DATASETS.get(dataset_name)
    if not dataset_path:
        raise ValueError(f"Dataset {dataset_name} not found in available datasets.")

    print(f"Downloading {dataset_name} dataset from Hugging Face...")
    dataset = load_dataset(dataset_path, trust_remote_code=True)
    if len(dataset['train']) > sample_size:
        dataset["train"] = dataset["train"].select(range(sample_size))
        print(len(dataset['train']))
    # Apply preprocessing
    print(f"Preprocessing {dataset_name} for {model_type} model...")
    dataset = dataset.map(lambda x: preprocess_text(x, model_type=model_type, dataset_name=dataset_name))

    # Save processed data
    save_path = os.path.join(PROCESSED_DIR, f"{dataset_name}_{model_type}_preprocessed.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Processed dataset saved at {save_path}")
    return dataset


# Example Usage
if __name__ == "__main__":
    # Load & preprocess for DistiGPT2
    tinystories_distilgpt2 = load_and_preprocess_dataset("tinystories", sample_size=10000, model_type="distilgpt2")
    fairytaleqa_distilgpt2 = load_and_preprocess_dataset("fairytaleqa", sample_size=10000, model_type="distilgpt2")
    # Load & preprocess for DistilBERT
    #tinystories_distilbert = load_and_preprocess_dataset("tinystories", model_type="distilbert")
    #fairytaleqa_distilbert = load_and_preprocess_dataset("fairytaleqa", model_type="distilbert")

    # Load & preprocess for LSTM
    # tinystories_lstm = load_and_preprocess_dataset("tinystories", model_type="lstm")
    # fairytaleqa_lstm = load_and_preprocess_dataset("fairytaleqa", model_type="lstm")
