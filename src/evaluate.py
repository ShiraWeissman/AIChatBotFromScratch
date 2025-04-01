import argparse
from datasets import load_from_disk

def load_chosen_model(task_type, model_type, model_path):
    if model_type == 'distilgpt2':
        from models.distilgpt2_model.model import load_model
        model = load_model(task_type, pretrained_model_name=model_path,
               pretrained_tokenizer_name="distilgpt2")
        return model
    elif model_type == 'distilbert':
        from models.distilbert_model.model import load_model
        model = load_model(task_type, pretrained_model_name=model_path)
        return model
    else:
        print("Error: Invalid model_type")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--model_type", type=str, choices=["distilbert", "distilgpt2"], required=True,
                        help="Type of model to evaluate")
    parser.add_argument("--task_type", type=str, choices=["language_modeling", "question_answering"], required=True,
                        help="Type of model to evaluate")
    args = parser.parse_args()
    print(f"Task type: {args.task_type}\n"
          f"Model type: {args.model_type}\n"
          f"Model path: {args.model_path}\n"
          f"Data path: {args.test_data_path}\n")
    print("Loading model..")
    model = load_chosen_model(task_type=args.task_type, model_type=args.model_type, model_path=args.model_path)
    print("Loading dataset..")
    data = load_from_disk(args.test_data_path)
    model.evaluate(data)