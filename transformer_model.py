from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class TextAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initialize the text analyzer with a pre-trained transformer model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode_text(self, text, max_length=128):
        """
        Encode text into a vector representation using the transformer model.
        """
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token
        return embeddings.flatten()

# Пример использования
if __name__ == "__main__":
    analyzer = TextAnalyzer()
    course_description = "This course covers advanced topics in machine learning."
    embedding = analyzer.encode_text(course_description)
    print("Text Embedding:", embedding)