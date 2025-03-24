import torch
import yaml
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.distilgpt2_model.model import load_model
from src.utils import root_path, load_config

def load_tokenizer(model_type="distilgpt2"):
    distilgpt2_tokenizer = AutoTokenizer.from_pretrained(model_type)
    distilgpt2_tokenizer.pad_token = distilgpt2_tokenizer.eos_token
    return distilgpt2_tokenizer

def generate_response(question, model, tokenizer, config):
    """
    Generate text using the trained DistilGPT-2 model.
    """
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)

    output_ids = model.generate(
        input_ids,
        max_length=config["max_length"],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=config["do_sample"],  # Enables randomness for more creative responses
        top_k=config["top_k"],  # Limits to top 50 words per step
        top_p=config["top_p"],  # Nucleus sampling
        temperature=config["temperature"]  # Controls randomness (lower = more deterministic)
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


if __name__ == '__main__':
    config = load_config("distilgpt2_inference_config")
    model = load_model(task_type="question_answering",
                       pretrained_model_name=os.path.join(root_path, config["model"]["path"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = load_tokenizer(model_type="distilgpt2")
    question = "What is the most common hero name?"
    response = generate_response(question, model, tokenizer, config)

    print("Question:", question)
    print("Generated Text:", response)


