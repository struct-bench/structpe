"""
Basic building blocks for the evaluator: LLMJudge, Verifier, and a simple Metrics container.
"""
import random

class LLMJudge:
    """
    Demo 'judge' that assigns a random 0..1 float score to a sample 
    (simulating an LLM-based evaluation).
    """
    def judge_sample(self, sample) -> float:
        return round(random.random(), 3)

class Verifier:
    """
    Basic text presence check. If sample has 'query_text', 'text', or 'paper_title',
    we require it be non-empty to consider it 'valid' text.
    """
    def verify_sample(self, sample) -> bool:
        text = getattr(sample, "query_text", None) or getattr(sample, "text", None) or getattr(sample, "paper_title", None)
        if text:
            return bool(text.strip())
        # If none of these attributes exist, return False
        return False

class Metrics:
    """
    Container for per-sample metric results.
      - score: the LLMJudge score
      - is_valid: whether constraints were met (or at least the sample wasn't flagged invalid).
    """
    def __init__(self, score: float, is_valid: bool):
        self.score = score
        self.is_valid = is_valid

    def to_dict(self):
        return {
            "score": self.score,
            "is_valid": self.is_valid
        }
