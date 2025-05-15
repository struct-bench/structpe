"""
_types.py: Common enumerations and atomic range classes 
used across various datasets.
"""

from enum import Enum

# For search queries
class QueryIntent(Enum):
    NAVIGATIONAL = "NAVIGATIONAL"
    TRANSACTIONAL = "TRANSACTIONAL"
    INFORMATIONAL = "INFORMATIONAL"
    RESEARCH = "RESEARCH"

class QueryTopic(Enum):
    TECHNOLOGY = "TECHNOLOGY"
    TRAVEL = "TRAVEL"
    EDUCATION = "EDUCATION"
    SHOPPING = "SHOPPING"

# For sentiment
class SentimentLabel(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

class AtomicRangeInt:
    """
    Represents an integer with a min/max range check.
    """
    def __init__(self, value: int, min_value: int, max_value: int):
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.verify_range()

    def verify_range(self):
        if not (self.min_value <= self.value <= self.max_value):
            raise ValueError(
                f"Integer {self.value} is out of range [{self.min_value}, {self.max_value}]."
            )

class AtomicRangeFloat:
    """
    Represents a float with a min/max range check.
    """
    def __init__(self, value: float, min_value: float, max_value: float):
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.verify_range()

    def verify_range(self):
        if not (self.min_value <= self.value <= self.max_value):
            raise ValueError(
                f"Float {self.value} out of range [{self.min_value}, {self.max_value}]."
            )
