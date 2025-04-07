
# Question Answering Project

This project implements a **Question Answering** chat bot using **Hugging Face** models and datasets. The goal of this project is to practice a simple AI chat bot application.
## Features

- **Model Selection**: Users can choose between two models:
  - **DistilGPT-2**: A distilled version of GPT-2, trained for text generation and question answering tasks.
  - **DistilBERT**: A distilled version of BERT, trained for question answering.
- **Context-Based Answers**: Users can input context and ask a question to get an answer based on the provided context.

## Technologies Used

- **Hugging Face Transformers**: For pre-trained models and tokenization.
- **Flask**: For creating the web application.
- **PyTorch**: For deep learning and model execution.

## Installation Instructions

To run this project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/ShiraWeissman/AIChatBotFromScratch.git
cd AIChatBotFromScratch
```

### 2. Install Dependencies

Create a virtual environment and install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

After installing the dependencies, start the Flask web server:

```bash
python app.py
```

The app should now be running at `http://127.0.0.1:5000/`.

## Usage

1. **Choose the Model**: Select the model (DistilGPT-2 or DistilBERT) from the dropdown menu.
2. **Input Context**: Provide context (e.g., a paragraph of text or a document).
3. **Ask a Question**: Enter your question related to the provided context.
4. **Generate Answer**: Click on the "Generate Answer" button to receive an answer.

## Available Training Notebooks

Training notebooks are available for both **DistilGPT-2** and **DistilBERT** models in this GitHub repository. These notebooks contain the code and instructions for training the models on your local machine or on Google Colab.

## Hugging Face Models

The following models have been trained and uploaded to Hugging Face:
- **DistilGPT-2 Language Model** and **DistilGPT-2 Question Answering Model**.
- **DistilBERT Language Model** and **DistilBERT Question Answering Model**.

You can download these models directly from Hugging Face using the following commands:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DistilBertConfig
from transformers import DistilBertForMaskedLM, DistilBertForQuestionAnswering, DistilBertTokenizerFast

# For DistilGPT-2
lm_config = AutoConfig.from_pretrained("ShiraWeis/distilgpt2-wikipedia-lm")
wikipedia_distilgpt2_lm_model = AutoModelForCausalLM.from_pretrained(lm_config)

qa_config = AutoConfig.from_pretrained("ShiraWeis/distilgpt2-trivia_qa-qa")
trivia_qa_distilgpt2_qa_model = AutoModelForCausalLM.from_pretrained(qa_config)

distilgpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# For DistilBERT
lm_config = DistilBertConfig.from_pretrained("ShiraWeis/distilbert-wikipedia-lm")
wikipedia_distilbert_lm_model = DistilBertForMaskedLM.from_pretrained(lm_config)

qa_config = DistilBertConfig.from_pretrained("ShiraWeis/distilbert-trivia_qa-qa")
trivia_qa_distilbert_qa_model = DistilBertForQuestionAnswering.from_pretrained(qa_config)

distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
```

## Training Data

- The **language models** (DistilGPT-2 and DistilBERT) were trained on a **subset of the Wikipedia dataset**.
- The **Question Answering models** (DistilGPT-2 QA and DistilBERT QA) were trained on the TriviaQA dataset**.

## License

This project is licensed under the **MIT License** - see the [MIT License](./LICENSE) file for details.

## Acknowledgements

- Hugging Face for providing pre-trained models and datasets.
- Google Colab for providing an easy-to-use platform for training and testing models.
- PyTorch for providing the framework for model deployment and inference.
- https://www.flaticon.com/ for the robot image.
- ChatGPT for useful advices and coding assistant.

