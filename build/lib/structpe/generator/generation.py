"""
generation.py: A simple example generator for the 'search_query' dataset.
It picks random words, intent, and topic, returns a JSON representation of the new sample.
"""

import json
import random

from structpe.descriptor.descriptor import DatasetDescriptor
from structpe._types import QueryIntent, QueryTopic, AtomicRangeInt
from structpe.dataset.search_dataset import SearchSample

class SampleGenerator:
    def __init__(self):
        self.descriptor = DatasetDescriptor()

    def generate_new_sample(self, serialized_dataset: str) -> str:
        """
        1) Deserializes the dataset (though we don't use it heavily here).
        2) Creates a new random search sample.
        3) Returns it as a JSON string.
        """
        _ = self.descriptor.deserialize(serialized_dataset)  # dataset is not heavily used here

        word_count_value = random.randint(2, 5)
        random_words = [random.choice(["ai", "travel", "python", "shop", "university"]) for _ in range(word_count_value)]
        query_text = " ".join(random_words)

        random_intent = random.choice(list(QueryIntent))
        random_topic = random.choice(list(QueryTopic))

        new_sample = SearchSample(
            query_text=query_text,
            intent=random_intent,
            topic=random_topic,
            word_count=AtomicRangeInt(word_count_value, 1, 10)
        )

        return json.dumps({
            "query_text": new_sample.query_text,
            "intent": new_sample.intent.value,
            "topic": new_sample.topic.value,
            "word_count": {
                "value": new_sample.word_count.value,
                "min_value": new_sample.word_count.min_value,
                "max_value": new_sample.word_count.max_value
            }
        })
