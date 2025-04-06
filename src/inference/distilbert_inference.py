import torch
from models.distilbert_model.model import load_model
from src.utils import root_path, load_config

def generate_response(context, question, model, config):
    """
    Generate text using the trained DistilGPT-2 model.
    """
    response = model.generate_answer(context, question, config)

    return response


if __name__ == '__main__':
    config = load_config("distilbert_inference_config")
    model = load_model(task_type="question_answering",
                       pretrained_model_name=config['model']['path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    context = "there was a prince named Or"
    question = "what is the name of the prince?"
    response = generate_response(context, question, model, config)
    print("Context:", context)
    print("Question:", question)
    print("Generated Text:", response)


