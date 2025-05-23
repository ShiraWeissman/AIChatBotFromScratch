import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertForMaskedLM, DistilBertForQuestionAnswering, DistilBertTokenizerFast
from torch.utils.data import DataLoader
from src.utils import *

class DistilBERTForLanguageModeling(nn.Module):
    """
    DistilBERT model for training on language modeling tasks (e.g., next-token prediction, masked language modeling).
    """

    def __init__(self, pretrained_model_name="distilbert-base-uncased", pretrained_tokenizer_name="distilbert-base-uncased"):
        super().__init__()
        self.model_type = "Language model"
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_tokenizer_name = pretrained_tokenizer_name
        if self.pretrained_model_name[-3:] == 'zip':
            print("Extracting zipped model..")
            extract_zipped_folder(os.path.join(root_path, self.pretrained_model_name))
            self.pretrained_model_name = os.path.join(root_path, self.pretrained_model_name.split('.')[0])
            print("Loading model..")
            self.model = DistilBertForMaskedLM.from_pretrained(self.pretrained_model_name)
        elif os.path.exists(os.path.join(root_path, self.pretrained_model_name)):
            self.pretrained_model_name = os.path.join(root_path, self.pretrained_model_name)
            print("Loading model..")
            self.model = DistilBertForMaskedLM.from_pretrained(self.pretrained_model_name)
        elif self.pretrained_model_name in ["distilbert-base-uncased", "ShiraWeis/distilbert-wikipedia-lm"]:
            print("Loading configuration...")
            self.config = DistilBertConfig.from_pretrained(self.pretrained_model_name)
            print("Loading model...")
            self.model = DistilBertForMaskedLM(self.config)
        else:
            print("Invalid pretrained_model_name")
            return
        print("Loading tokenizer...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.pretrained_tokenizer_name)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def evaluate(self, dataset, batch_size=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Evaluates the model on a given dataset and returns the perplexity.
        """
        self.model.to(device)
        self.model.eval()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                total_loss += loss.item() * input_ids.size(0)
                total_tokens += torch.sum(attention_mask).item()

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))

        return perplexity.item()

    def save_model(self, save_path="models/distilbert_lm"):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        zip_folder(save_path)


class DistilBERTForQuestionAnswering(nn.Module):
    """
    DistilBERT model for extractive question answering tasks (e.g., FairytaleQA).
    Predicts start and end tokens for answer spans.
    """
    def __init__(self, pretrained_model_name="distilbert-base-uncased", pretrained_tokenizer_name="distilbert-base-uncased"):
        super().__init__()

        self.model_type = "Question Answering model"
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_tokenizer_name = pretrained_tokenizer_name
        print("Loading model...")
        if self.pretrained_model_name[-3:] == 'zip':
            print("Extracting zipped model..")
            extract_zipped_folder(os.path.join(root_path, self.pretrained_model_name))
            self.pretrained_model_name = os.path.join(root_path, self.pretrained_model_name.split('.')[0])
            print("Loading model..")
            self.model = DistilBertForQuestionAnswering.from_pretrained(self.pretrained_model_name)
        elif os.path.exists(os.path.join(root_path, self.pretrained_model_name)):
            self.pretrained_model_name = os.path.join(root_path, self.pretrained_model_name)
            self.pretrained_tokenizer_name = pretrained_model_name
            print("Loading model..")
            self.model = DistilBertForQuestionAnswering.from_pretrained(self.pretrained_model_name)
        elif self.pretrained_model_name in ["distilbert-base-uncased", "ShiraWeis/distilbert-trivia_qa-qa"]:
            print("Loading configuration...")
            self.config = DistilBertConfig.from_pretrained(self.pretrained_model_name)
            print("Loading model...")
            self.model = DistilBertForQuestionAnswering(self.config)
        else:
            print("Invalid pretrained_model_name")
            return

        print("Loading tokenizer...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.pretrained_tokenizer_name)
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        return outputs

    def save_model(self, save_path="models/distilbert_qa"):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        zip_folder(save_path)

    def evaluate(self, dataset, batch_size=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Evaluates the model on a given question-answer dataset.
        Computes loss as a measure of performance.
        """
        self.model.to(device)
        self.model.eval()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    import torch

    def generate_answer(self, question, context):
        """
        Generates an answer given a context and a question using DistilBERT for Question Answering.
        Ensures that predicted answer spans are only from the context part.
        """
        encoding = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
            return_token_type_ids=True
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]  # 0 for question, 1 for context
        offset_mapping = encoding["offset_mapping"][0]  # (start_char, end_char) per token

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Mask out tokens that are NOT in the context (token_type_id == 1 keeps only context)
        context_mask = token_type_ids[0] == 1
        invalid_mask = ~context_mask

        start_logits[invalid_mask] = float('-inf')
        end_logits[invalid_mask] = float('-inf')

        # Get most probable start and end indices (within context)
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item() + 1

        # Use offset_mapping to ensure start and end indices fall within the context span
        # (check if the character span of the answer is within the context)
        while token_type_ids[0][start_idx] != 1 and start_idx > 0:
            start_idx -= 1
        while token_type_ids[0][end_idx - 1] != 1 and end_idx < len(input_ids[0]):
            end_idx += 1

        # Get the character offsets for the predicted tokens
        start_char, _ = offset_mapping[start_idx]
        _, end_char = offset_mapping[end_idx - 1]

        # Ensure the span falls within the context (based on character indices)
        if start_char < len(context) and end_char <= len(context):
            answer_tokens = input_ids[0][start_idx:end_idx]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        else:
            answer = "I don't know."

        return answer.strip() if answer.strip() else "I don't know."


def load_model(task_type="language_modeling", pretrained_model_name="distilbert-base-uncased",
               pretrained_tokenizer_name="distilbert-base-uncased"):
    """
    Load the appropriate DistilBERT model for the given task.

    Args:
        task_type (str): "language_modeling" or "question_answering"
        model_name (str): Pretrained model name
        pretrained_tokenizer_name (str): pretrained tokenizer name

    Returns:
        model (nn.Module): Loaded model instance
    """
    if task_type.lower() == "language_modeling":
        model = DistilBERTForLanguageModeling(pretrained_model_name)
    elif task_type.lower() == "question_answering":
        model = DistilBERTForQuestionAnswering(pretrained_model_name)
    else:
        raise ValueError("Invalid task_type. Choose 'language_modeling' or 'question_answering'.")

    return model


if __name__ == '__main__':
    pass