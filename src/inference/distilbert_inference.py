import torch
from models.distilbert_model.model import load_model
from src.utils import root_path, load_config

def generate_response(question, context,  model):
    """
    Generate text using the trained DistilGPT-2 model.
    """
    response = model.generate_answer(question, context)

    return response


if __name__ == '__main__':
    config = load_config("distilbert_inference_config")
    model = load_model(task_type="question_answering",
                       pretrained_model_name=config['model']['path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    context = "The ingredients for fruit salad are: apple, banana, strawberry."#"The capital of France is Paris"
    question = "Which fruits are in fruit salad?"#"what is the capital of France"
    response = generate_response(question, context, model)
    print("Context:", context)
    print("Question:", question)
    print("Generated Text:", response)


