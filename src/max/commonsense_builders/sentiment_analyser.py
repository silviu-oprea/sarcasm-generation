import csv
import numpy as np
import urllib.request

import torch

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax


TASK = "sentiment"
MODEL = f"cardiffnlp/twitter-roberta-base-{TASK}"
LABEL_MAPPING_LINK = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{TASK}/mapping.txt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def preprocess_text(text):
    """Preprocess the text. For now, replace handles with "@user" and links
    with "http".
    """
    processed_tokens = []
    for token in text.split():
        if token.startswith('@') and len(token) > 1:
            processed_tokens.append("@user")
        elif token.startswith("http"):
            processed_tokens.append("http")
        else:
            processed_tokens.append(token)
    return " ".join(processed_tokens)


class SentimentAnalyser:
    def __init__(self, model, tokenizer, labels):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels

    @classmethod
    def default(cls):
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL
        ).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        with urllib.request.urlopen(LABEL_MAPPING_LINK) as f:
            html = f.read().decode('utf-8').split("\n")
            reader = csv.DictReader(
                html, delimiter='\t', fieldnames=["index", "polarity"]
            )
        labels = [row["polarity"] for row in reader if len(row) > 1]
        return cls(model, tokenizer, labels)

    def get_sentiment(self, text, excluded=['neutral']):
        text = preprocess_text(text)
        encoded_input = self.tokenizer(text, return_tensors='pt').to(DEVICE)
        output = self.model(**encoded_input)
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)[::-1]
        for rank in ranking:
            if self.labels[rank] not in excluded:
                return self.labels[rank]

    def get_sentiment_dist(self, text):
        text = preprocess_text(text)
        encoded_input = self.tokenizer(text, return_tensors='pt').to(DEVICE)
        output = self.model(**encoded_input)
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)[::-1]
        return [(self.labels[rank], scores[rank]) for rank in ranking]


if __name__ == "__main__":
    analyser = SentimentAnalyser.default()
    while True:
        text = input('Enter text: ')
        print(text, analyser.get_sentiment(text))
