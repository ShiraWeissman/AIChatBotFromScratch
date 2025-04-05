from datasets import load_dataset
from transformers import DistilBertTokenizer, AutoTokenizer
from src.utils import *

DATASETS = {
    "wikipedia": "MedRAG/wikipedia",
    "trivia_qa": "mandarjoshi/trivia_qa"
}

processed_dir = os.path.join(root_path, "data/processed/")
os.makedirs(processed_dir, exist_ok=True)

distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

distilgpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
distilgpt2_tokenizer.pad_token = distilgpt2_tokenizer.eos_token

def check_existing_files(model_type, dataset_name):
    file_path = os.path.join(processed_dir, f"{dataset_name}_{model_type}_preprocessed.pkl")
    return file_path if os.path.exists(file_path) else None


def preprocess_text(raw, model_type="distilgpt2", dataset_name="wikipedia"):
    """Preprocesses text for the chosen model type (DistilBERT for QA or DistilGPT-2 for generation).
    Supports Wikipedia (for LM) and TriviaQA (for QA).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "distilbert":
        if dataset_name == "wikipedia":
            tokenized_output = distilbert_tokenizer(
                raw["content"], padding="max_length", truncation=True, max_length=512
            )
            tokenized_output = {
                key: torch.tensor(val).to(device) for key, val in tokenized_output.items()
            }
            tokenized_output["labels"] = tokenized_output["input_ids"]
            return tokenized_output

        elif dataset_name == "trivia_qa":
            # TriviaQA for extractive QA with DistilBERT
            question = raw["question"]
            context = raw["search_results"]["search_context"] if "search_results" in raw else raw.get("context", "")
            answer = raw["answer"]["value"] if isinstance(raw["answer"], dict) else raw["answer"]

            tokenized_output = distilbert_tokenizer(
                context, question, padding="max_length", truncation=True, max_length=512
            )
            answer_tokens = distilbert_tokenizer(answer, add_special_tokens=False)["input_ids"]
            context_tokens = tokenized_output["input_ids"]

            start_position, end_position = 0, 0
            for i in range(len(context_tokens) - len(answer_tokens) + 1):
                if context_tokens[i:i + len(answer_tokens)] == answer_tokens:
                    start_position, end_position = i, i + len(answer_tokens) - 1
                    break

            tokenized_output["start_positions"] = torch.tensor(start_position).to(device)
            tokenized_output["end_positions"] = torch.tensor(end_position).to(device)
            tokenized_output = {
                key: torch.tensor(val).to(device) if not torch.is_tensor(val) else val
                for key, val in tokenized_output.items()
            }
            return tokenized_output

    elif model_type == "distilgpt2":
        if dataset_name == "wikipedia":
            # Wikipedia for DistilGPT2 language modeling
            tokenized_output = distilgpt2_tokenizer(
                raw["content"], truncation=True, padding="max_length", max_length=256
            )
            tokenized_output["labels"] = tokenized_output["input_ids"]
            tokenized_output = {
                key: torch.tensor(val).to(device) for key, val in tokenized_output.items()
            }
            return tokenized_output

        elif dataset_name == "trivia_qa":
            question = raw["question"]
            context = raw["search_results"]["search_context"] if "search_results" in raw else raw.get("context", "")
            answer = raw["answer"]["value"] if isinstance(raw["answer"], dict) else raw["answer"]

            input_text = f"Context: {context} Question: {question} Answer: {answer}"
            tokenized_output = distilgpt2_tokenizer(
                input_text, truncation=True, padding="max_length", max_length=512
            )
            tokenized_output["labels"] = tokenized_output["input_ids"]
            tokenized_output = {
                key: torch.tensor(val).to(device) for key, val in tokenized_output.items()
            }
            return tokenized_output



def load_and_preprocess_dataset(dataset_name, model_type="distilgpt2", sample_size=10000, force_reprocess=False):
    """
    Loads the dataset from Hugging Face and preprocesses it for the specified model type.
    :param dataset_name: Name of the dataset (e.g., 'wikipedia' or 'trivia_qa').
    :param model_type: 'distilgpt2' or 'distilbert'.
    :param force_reprocess: If True, forces reprocessing even if a saved file exists.
    :return: Preprocessed dataset.
    """
    dataset_name = dataset_name.lower()
    processed_file = check_existing_files(model_type, dataset_name)

    if processed_file and not force_reprocess:
        print(f"Loading preprocessed {dataset_name} dataset from {processed_file}...")
        with open(processed_file, "rb") as f:
            return pickle.load(f)

    dataset_path = DATASETS.get(dataset_name)
    if not dataset_path:
        raise ValueError(f"Dataset {dataset_name} not found in available datasets.")

    print(f"Downloading {dataset_name} dataset from Hugging Face...")
    if dataset_name == 'trivia_qa':
        dataset = load_dataset(dataset_path,  "unfiltered", trust_remote_code=True)
    elif dataset_name == 'wikipedia':
        dataset = load_dataset(dataset_path,  trust_remote_code=True)
    if len(dataset['train']) > sample_size:
        dataset["train"] = dataset["train"].select(range(sample_size))
        print(len(dataset['train']))

    print(f"Preprocessing {dataset_name} for {model_type} model...")
    dataset = dataset.map(lambda x: preprocess_text(x, model_type=model_type, dataset_name=dataset_name))

    save_path = os.path.join(processed_dir, f"{dataset_name}_{model_type}_preprocessed")
    dataset.save_to_disk(save_path)

    print(f"Processed dataset saved at {save_path}")
    return dataset



if __name__ == "__main__":
    # Load & preprocess for DistiGPT2
    # wikipedia_distilgpt2 = load_and_preprocess_dataset("wikipedia", sample_size=10000,
    #                                                      model_type="distilgpt2", force_reprocess=True)
    # trivia_qa_distilgpt2 = load_and_preprocess_dataset("trivia_qa", sample_size=10000,
    #                                                      model_type="distilgpt2", force_reprocess=True)
    # Load & preprocess for DistilBERT
    # wikipedia_distilbert = load_and_preprocess_dataset("wikipedia", sample_size=30,
    #                                                      model_type="distilbert", force_reprocess=True)
    trivia_qa_distilbert = load_and_preprocess_dataset("trivia_qa", sample_size=30,
                                                         model_type="distilbert", force_reprocess=True)

