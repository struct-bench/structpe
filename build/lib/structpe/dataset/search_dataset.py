"""
Dataset for search queries, storing query text, intent, topic, and word_count.
"""

from structpe._types import QueryIntent, QueryTopic, AtomicRangeInt
from structpe.dataset.registry import register_dataset

class SearchSample:
    def __init__(self, query_text: str, intent: QueryIntent, topic: QueryTopic, word_count: AtomicRangeInt):
        """
        query_text: The actual search string
        intent: An enum (NAVIGATIONAL, TRANSACTIONAL, etc.)
        topic: Another enum (TECHNOLOGY, TRAVEL, etc.)
        word_count: AtomicRangeInt specifying the # of words
        """
        self.query_text = query_text
        self.intent = intent
        self.topic = topic
        self.word_count = word_count
        self.verify()

    def verify(self):
        actual_count = len(self.query_text.split())
        if actual_count != self.word_count.value:
            raise ValueError(
                f"Word count mismatch: declared={self.word_count.value}, actual={actual_count}."
            )

class SearchDataset:
    def __init__(self):
        self.samples = []

    def add_sample(self, sample: SearchSample):
        """
        Adds a search sample to the dataset after verifying it.
        """
        sample.verify()
        self.samples.append(sample)

    def verify_all(self):
        """
        Re-verify all samples in the dataset.
        """
        for s in self.samples:
            s.verify()

register_dataset("search_query", SearchDataset)
