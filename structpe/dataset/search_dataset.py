"""
Dataset for search queries, storing query text, intent, topic, and word_count.
"""
from structpe._types import QueryIntent, QueryTopic, AtomicRangeInt
from structpe.dataset.registry import register_dataset

class SearchSample:
    def __init__(self, query_text: str, intent: QueryIntent, topic: QueryTopic, word_count: AtomicRangeInt):
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
        sample.verify()
        self.samples.append(sample)

    def verify_all(self):
        for s in self.samples:
            s.verify()

def get_prompt() -> str:
    """
    Return a triple-quoted multiline prompt for the search query generator.
    """
    return """You are an AI creating random search queries.
Please generate queries that a typical user might input into a search engine.
Use a creative variety of search terms, but keep them somewhat realistic."""

def get_variation() -> str:
    """
    Return a triple-quoted multiline variation text.
    """
    return """Rewrite the following search query, preserving the intent,
while rephrasing or substituting synonyms. The user wants an alternative query
with the same meaning but different wording or keywords.
""" 

register_dataset("search_query", SearchDataset)