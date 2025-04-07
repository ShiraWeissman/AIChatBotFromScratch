import os
from flask import Flask, request, jsonify, render_template
import torch
from src.utils import load_config, root_path

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_chosen_config(model_choice):
    if model_choice == "distilgpt2":
        config = load_config("distilgpt2_inference_config")
    elif model_choice == "distilbert":
        config = load_config("distilbert_inference_config")
    return config


@app.route("/")
def home():
    return render_template("index.html", model=None)

def load_chosen_model(model_choice, config):
    """Dynamically loads the requested model."""
    if model_choice == "distilgpt2":
        from models.distilgpt2_model.model import load_model
        return load_model(task_type="question_answering", pretrained_model_name=config['model']['path']).to(device)
    elif model_choice == "distilbert":
        from models.distilbert_model.model import load_model
        return load_model(task_type="question_answering", pretrained_model_name=config['model']['path']).to(device)
    else:
        return None

@app.route("/generate_answer", methods=["POST"])
def generate_answer():
    try:
        model_choice = request.form.get("model")  # No default value to force user choice
        context = request.form.get("context")
        question = request.form.get("question")

        if not model_choice or model_choice not in ['distilgpt2', 'distilbert']:
            return jsonify({"error": "Invalid or missing model choice"}), 400

        if not context or not question:
            return jsonify({"error": "Context or question missing"}), 400

        config = load_chosen_config(model_choice)
        model = load_chosen_model(model_choice, config)
        if not model:
            return jsonify({"error": "Failed to load model"}), 500
        if model_choice == 'distilgpt2':
            answer = model.generate_answer(context, question, config)
        elif model_choice == 'distilbert':
            answer = model.generate_answer(context, question)

        return render_template("index.html", model=model_choice, context=context, question=question, answer=answer)

    except Exception as e:
        print("Error in /generate_answer:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
