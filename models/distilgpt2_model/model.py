from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import *


class DistilGPT2ForLanguageModeling(nn.Module):
    """
    Custom class for training a DistilGPT-2 model with pre-tokenized data.
    """

    def __init__(self, pretrained_model_name="distilgpt2", pretrained_tokenizer_name="distilgpt2"):
        super().__init__()
        self.model_type = "Language model"
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_tokenizer_name = pretrained_tokenizer_name

        if self.pretrained_model_name[-3:] == 'zip':
            print("Extracting zipped model..")
            extract_zipped_folder(os.path.join(root_path, self.pretrained_model_name))
            self.pretrained_model_name = os.path.join(root_path, self.pretrained_model_name.split('.')[0])
            print("Loading model..")
            self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name)
        elif os.path.exists(os.path.join(root_path, self.pretrained_model_name)):
            self.pretrained_model_name = os.path.join(root_path, self.pretrained_model_name)
            print("Loading model..")
            self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name)
        elif self.pretrained_model_name in ["distilgpt2", "ShiraWeis/distilgpt2-wikipedia-lm"]:
            print("Loading configuration...")
            self.config = AutoConfig.from_pretrained(self.pretrained_model_name)
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_config(self.config)
        else:
            print("Invalid pretrained_model_name")
            return
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_tokenizer_name)
        print("Setting padding token...")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def evaluate(self, dataset, batch_size=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Evaluates the model on a given dataset and returns the perplexity.
        """
        self.model.to(device)
        self.model.eval()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                total_loss += loss.item() * input_ids.size(0)
                total_tokens += torch.sum(attention_mask).item()

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))

        return perplexity.item()


    def save_model(self, save_path="models/distilgpt2_lm"):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        zip_folder(save_path)


class DistilGPT2ForQuestionAnswering(nn.Module):
    """
    Fine-tuned DistilGPT-2 for Question Answering.
    Uses causal language modeling to generate answers based on a question and context.
    """

    def __init__(self, pretrained_model_name="distilgpt2", pretrained_tokenizer_name="distilgpt2"):
        super().__init__()

        self.model_type = "Question Answering model"
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_tokenizer_name = pretrained_tokenizer_name
        print("Loading model...")
        if self.pretrained_model_name[-3:] == 'zip':
            print("Extracting zipped model..")
            extract_zipped_folder(os.path.join(root_path, self.pretrained_model_name))
            self.pretrained_model_name = os.path.join(root_path, self.pretrained_model_name.split('.')[0])
            print("Loading model..")
            self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name)
        elif os.path.exists(os.path.join(root_path, self.pretrained_model_name)):
            self.pretrained_model_name = os.path.join(root_path, self.pretrained_model_name)
            self.pretrained_tokenizer_name = pretrained_model_name
            print("Loading model..")
            self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name)
        elif self.pretrained_model_name in ["distilgpt2", "ShiraWeis/distilgpt2-trivia_qa-qa"]:
            print("Loading configuration...")
            self.config = AutoConfig.from_pretrained(self.pretrained_model_name)
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_config(self.config)
        else:
            print("Invalid pretrained_model_name")
            return

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)

        print("Setting padding token...")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(self, dataset, batch_size=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Evaluates the model on a given question-answer dataset.
        Computes loss as a measure of performance.
        """
        self.model.to(device)
        self.model.eval()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    def generate_answer(self, context, question, config):
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
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if 'answer' in response.lower():
            response = response.split('Answer:')[-1].strip()
        return response

    def save_model(self, save_path="models/distilgpt2_qa"):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        zip_folder(save_path)


def load_model(task_type="language_modeling", pretrained_model_name="distilgpt2",
               pretrained_tokenizer_name="distilgpt2"):
    """
    Load the appropriate Distilgpt2 model for the given task.

    Args:
        task_type (str): "language_modeling" or "question_answering"
        model_name (str): Pretrained model name
    Returns:
        model (): Loaded model instance
    """
    if task_type.lower() == "language_modeling":
        print("Loading DistilGPT2ForLanguageModeling..")
        model = DistilGPT2ForLanguageModeling(pretrained_model_name, pretrained_tokenizer_name)
    elif task_type.lower() == "question_answering":
        print("Loading DistilGPT2ForQuestionAnswering..")
        model = DistilGPT2ForQuestionAnswering(pretrained_model_name, pretrained_tokenizer_name)
    else:
        raise ValueError("Invalid task_type. Choose 'language_modeling' or 'question_answering'.")

    return model
