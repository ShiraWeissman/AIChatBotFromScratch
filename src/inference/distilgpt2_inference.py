import torch
import yaml
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.distilgpt2_model.model import load_model
from src.utils import root_path, load_config

# def load_tokenizer(model_type="distilgpt2"):
#     distilgpt2_tokenizer = AutoTokenizer.from_pretrained(model_type)
#     distilgpt2_tokenizer.pad_token = distilgpt2_tokenizer.eos_token
#     return distilgpt2_tokenizer

def generate_response(question, model, config):
    """
    Generate text using the trained DistilGPT-2 model.
    """
    response = model.generate_answer(question, config, context="Fairy tales",  max_length=50)

    return response


if __name__ == '__main__':
    config = load_config("distilgpt2_inference_config")
    model = load_model(task_type="question_answering",
                       pretrained_model_name='distilgpt2') #os.path.join(root_path, config["model"]["path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    question = "What is the most common hero name?"
    response = generate_response(question, model, config)

    print("Question:", question)
    print("Generated Text:", response)


