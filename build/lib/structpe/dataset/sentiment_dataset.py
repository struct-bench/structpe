"""
A simple sentiment dataset with text + sentiment label (POSITIVE, NEGATIVE, NEUTRAL).
"""

from structpe._types import SentimentLabel
from structpe.dataset.registry import register_dataset

class SentimentSample:
    def __init__(self, text: str, sentiment: SentimentLabel):
        self.text = text
        self.sentiment = sentiment
        self.verify()

    def verify(self):
        if not self.text.strip():
            raise ValueError("Text is empty; invalid sample.")

class SentimentDataset:
    def __init__(self):
        self.samples = []

    def add_sample(self, sample: SentimentSample):
        sample.verify()
        self.samples.append(sample)

    def verify_all(self):
        for s in self.samples:
            s.verify()

register_dataset("sentiment", SentimentDataset)
