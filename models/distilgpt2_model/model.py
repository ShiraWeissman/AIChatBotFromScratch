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

    def save_model(self, save_path="models/distilgpt2_lm"):
        """ Save the trained model and tokenizer correctly """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, model_path="models/distilgpt2_lm"):
        """ Load the trained model and tokenizer """
        print("Loading trained model from:", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)  # Load model correctly
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


class DistilGPT2ForQuestionAnswering(nn.Module):
    """
    Fine-tuned DistilGPT-2 for Question Answering.
    Uses causal language modeling to generate answers based on a question and context.
    """

    def __init__(self, pretrained_model_name="distilgpt2"):
        super().__init__()  # Fix: Initialize nn.Module correctly

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        print("Setting padding token...")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 doesn’t have a pad token

    def generate_answer(self, question, config, context="Fairy tales", max_length=50):
        """
        Generates an answer given a context and a question.
        """
        input_text = f"Context: {context} Question: {question} Answer:"

        encoding = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask

        output_ids = self.model.generate(input_ids,
                                         attention_mask=attention_mask,
                                         max_length=config["inference"]["max_length"],
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         do_sample=config["inference"]["do_sample"],
                                         top_k=config["inference"]["top_k"],
                                         top_p=config["inference"]["top_p"],
                                         temperature=config["inference"]["temperature"])
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def save_model(self, save_path="models/distilgpt2_qa"):
        """ Saves the trained model and tokenizer correctly """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, model_path="models/distilgpt2_qa"):
        """ Loads a fine-tuned model and tokenizer correctly """
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path)  # Fix: Correct loading method
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    # def generate_answer(self, context, question, max_length=50):
    #     """
    #     Generates an answer given a context and a question.
    #     """
    #     input_text = f"Context: {context} Question: {question} Answer:"
    #     input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
    #
    #     output_ids = self.model.generate(input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
    #     return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # def generate(self, prompt, max_length=50, do_sample=True, top_k=50, top_p=0.95, temperature=0.7):
    #     """
    #     Generates text based on the provided prompt using causal language modeling.
    #     This can be used for more general text generation (e.g., story generation, creative text, etc.).
    #     """
    #     # Tokenize the prompt and create an attention mask
    #     encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    #     input_ids = encoding.input_ids.to(self.model.device)  # Ensure input_ids are on the correct device
    #     attention_mask = encoding.attention_mask.to(self.model.device)  # Attention mask for padding tokens
    #
    #     # Generate text using the model
    #     output_ids = self.model.generate(
    #         input_ids,
    #         attention_mask=attention_mask,  # Pass attention mask
    #         max_length=max_length,
    #         pad_token_id=self.tokenizer.eos_token_id,
    #         do_sample=do_sample,  # Enables randomness for creative text
    #         top_k=top_k,  # Limits to top K words per step
    #         top_p=top_p,  # Nucleus sampling
    #         temperature=temperature  # Controls randomness (lower = more deterministic)
    #     )
    #
    #     # Convert tensor to list for decoding
    #     output_ids = output_ids[0].cpu().numpy().tolist()
    #
    #     # Decode the output_ids to a string
    #     output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
    #     return output_text


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
        model = DistilGPT2ForQuestionAnswering(pretrained_model_name)
        # model = DistilGPT2ForQuestionAnswering(pretrained_model_name)
    else:
        raise ValueError("Invalid task_type. Choose 'language_modeling' or 'question_answering'.")

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
        print(f"Loaded model weights from {checkpoint_path}")

    return model
