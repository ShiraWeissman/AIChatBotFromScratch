import torch
from models.distilgpt2_model.model import load_model
from src.utils import root_path, load_config

def generate_response(context, question, model, config):
    """
    Generate text using the trained DistilGPT-2 model.
    """
    response = model.generate_answer(context, question, config)

    return response


if __name__ == '__main__':
    config = load_config("distilgpt2_inference_config")
    model = load_model(task_type="question_answering",
                       pretrained_model_name=config['model']['path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    context = ''
    question = "What is the most common hero name?"
    response = generate_response(context, question, model, config)

    print("Question:", question)
    print("Generated Text:", response)


