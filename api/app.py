import os
from flask import Flask, request, jsonify, render_template
import torch
from src.utils import load_config, root_path
from models.distilgpt2_model.model import DistilGPT2ForQuestionAnswering

app = Flask(__name__)

# Load configuration
config = load_config("distilgpt2_inference_config")

# Initialize the model with configuration
model = DistilGPT2ForQuestionAnswering(pretrained_model_name=config["model"]["path"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate_answer", methods=["POST"])
def generate_answer():
    try:
        context = request.form.get("context")
        question = request.form.get("question")

        if not context or not question:
            return jsonify({"error": "Context or question missing"}), 400

        answer = model.generate_answer(question, config, context=context)
        return render_template("index.html", context=context, question=question, answer=answer)
    except Exception as e:
        print("Error in /generate_answer:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
