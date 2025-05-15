"""
test_evaluator.py: Tests for the Evaluator's logic.
We create a dataset, run evaluation, and check that the output has expected keys.
"""

import unittest
from structpe.evaluator.evaluator import Evaluator
from structpe.dataset.search_dataset import SearchDataset, SearchSample
from structpe._types import QueryIntent, QueryTopic, AtomicRangeInt

class TestEvaluator(unittest.TestCase):
    def test_evaluator_search_dataset(self):
        ds = SearchDataset()
        sample = SearchSample("hello world", QueryIntent.NAVIGATIONAL, QueryTopic.TRAVEL, AtomicRangeInt(2,1,10))
        ds.add_sample(sample)

        evaluator = Evaluator()
        # We won't write to a file here, just get the dict results
        metrics = evaluator.evaluate_and_write_json(ds, output_json=None)

        self.assertIn("total_samples", metrics)
        self.assertIn("valid_count", metrics)
        self.assertIn("invalid_count", metrics)
        self.assertIn("invalid_samples", metrics)
        self.assertIn("distribution", metrics)
        self.assertIn("sample_level_results", metrics)
        self.assertIn("average_llm_score", metrics)

if __name__ == "__main__":
    unittest.main()
