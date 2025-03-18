import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertForMaskedLM, DistilBertForQuestionAnswering


class DistilBERTForLanguageModeling(nn.Module):
    """
    DistilBERT model for training on language modeling tasks (e.g., next-token prediction, masked language modeling).
    """

    def __init__(self, model_name="distilbert-base-uncased"):
        super(DistilBERTForLanguageModeling, self).__init__()
        self.config = DistilBertConfig.from_pretrained(model_name)
        self.model = DistilBertForMaskedLM.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits  # Loss for training, logits for prediction


class DistilBERTForQuestionAnswering(nn.Module):
    """
    DistilBERT model for extractive question answering tasks (e.g., FairytaleQA).
    Predicts start and end tokens for answer spans.
    """

    def __init__(self, model_name="distilbert-base-uncased"):
        super(DistilBERTForQuestionAnswering, self).__init__()
        self.config = DistilBertConfig.from_pretrained(model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        return outputs.loss, outputs.start_logits, outputs.end_logits  # Loss and logits


def load_model(task_type="language_modeling", model_name="distilbert-base-uncased", checkpoint_path=None):
    """
    Load the appropriate DistilBERT model for the given task.

    Args:
        task_type (str): "language_modeling" or "question_answering"
        model_name (str): Pretrained model name
        checkpoint_path (str, optional): Path to checkpoint file

    Returns:
        model (nn.Module): Loaded model instance
    """
    if task_type.lower() == "language_modeling":
        model = DistilBERTForLanguageModeling(model_name)
    elif task_type.lower() == "question_answering":
        model = DistilBERTForQuestionAnswering(model_name)
    else:
        raise ValueError("Invalid task_type. Choose 'language_modeling' or 'question_answering'.")

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
        print(f"Loaded model weights from {checkpoint_path}")

    return model


if __name__ == '__main__':
    pass