import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering
from src.utils import *


class DistilGPT2ForLanguageModeling(nn.Module):
    """
    Custom class for training a DistilGPT-2 model with pre-tokenized data.
    """

    def __init__(self, pretrained_model_name="distilgpt2"):
        super().__init__()  # Inherit from nn.Module, not AutoModelForCausalLM

        print("Loading configuration...")
        self.config = AutoConfig.from_pretrained(pretrained_model_name)

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_config(self.config)

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        print("Setting padding token...")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 doesn't have a pad token by default

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def save_model(self, save_path="models/distilgpt2_model"):
        """ Save the trained model and tokenizer correctly """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, model_path="models/distilgpt2_model"):
        """ Load the trained model and tokenizer """
        print("Loading trained model from:", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)  # Load model correctly
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

class DistilGPT2ForQuestionAnswering(AutoModelForQuestionAnswering):
    """
    Fine-tuned DistilGPT-2 for Question Answering.
    Uses causal language modeling to generate answers based on a question and context.
    """

    def __init__(self, pretrained_model_name="distilgpt2"):
        self.model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 doesn’t have a pad token

    def generate_answer(self, context, question, max_length=50):
        """
        Generates an answer given a context and a question.
        """
        input_text = f"Context: {context} Question: {question} Answer:"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        output_ids = self.model.generate(input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def save_model(self, save_path="models/distilgpt2_qa"):
        """ Saves the trained model and tokenizer correctly """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self):
        self.model.load_pretrained(self.pretrinaed_model_path)
        self.tokenizer.load_pretrained(self.pretrinaed_tokenizer_path)


def load_model(task_type="language_modeling", pretrained_model_name="distilgpt2", checkpoint_path=None):
    """
    Load the appropriate Distilgpt2 model for the given task.

    Args:
        task_type (str): "language_modeling" or "question_answering"
        model_name (str): Pretrained model name
        checkpoint_path (str, optional): Path to checkpoint file

    Returns:
        model (): Loaded model instance
    """
    if task_type.lower() == "language_modeling":
        print("Loading DistilGPT2ForLanguageModeling..")
        model = DistilGPT2ForLanguageModeling(pretrained_model_name)
    elif task_type.lower() == "question_answering":
        print("Loading DistilGPT2ForQuestionAnswering..")
        model = DistilGPT2ForQuestionAnswering(os.path.join(root_path, "models/distilgpt2_lm"))
        #model = DistilGPT2ForQuestionAnswering(pretrained_model_name)
    else:
        raise ValueError("Invalid task_type. Choose 'language_modeling' or 'question_answering'.")

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
        print(f"Loaded model weights from {checkpoint_path}")

    return model