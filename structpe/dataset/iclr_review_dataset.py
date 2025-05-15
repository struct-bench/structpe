"""
A dataset modeling ICLR paper reviews, with rating, confidence, recommendation, and multiple comments.
"""

from enum import Enum
from structpe._types import AtomicRangeInt, AtomicRangeFloat
from structpe.dataset.registry import register_dataset

class ICLRCommentRole(Enum):
    REVIEWER = "REVIEWER"
    AUTHOR = "AUTHOR"
    AREA_CHAIR = "AREA_CHAIR"

class ICLRComment:
    """
    A single comment in the discussion thread for a paper.
    """
    def __init__(self, text: str, role: ICLRCommentRole):
        self.text = text
        self.role = role
        self.verify()

    def verify(self):
        if not self.text.strip():
            raise ValueError("ICLRComment: text is empty, invalid comment.")

class ICLRRecommendation(Enum):
    STRONG_ACCEPT = "STRONG_ACCEPT"
    WEAK_ACCEPT = "WEAK_ACCEPT"
    BORDERLINE = "BORDERLINE"
    WEAK_REJECT = "WEAK_REJECT"
    STRONG_REJECT = "STRONG_REJECT"

class ICLRPaperReviewSample:
    """
    A single review for a paper:
     - paper_title
     - rating: (AtomicRangeInt)
     - confidence: (AtomicRangeFloat)
     - recommendation: (ICLRRecommendation enum)
     - comments: list of ICLRComment
    """
    def __init__(
        self,
        paper_title: str,
        rating: AtomicRangeInt,
        confidence: AtomicRangeFloat,
        recommendation: ICLRRecommendation,
        comments: list[ICLRComment]
    ):
        self.paper_title = paper_title
        self.rating = rating
        self.confidence = confidence
        self.recommendation = recommendation
        self.comments = comments
        self.verify()

    def verify(self):
        if not self.paper_title.strip():
            raise ValueError("ICLRPaperReviewSample: paper title is empty.")
        for c in self.comments:
            c.verify()

class ICLRReviewDataset:
    def __init__(self):
        self.samples = []

    def add_sample(self, sample: ICLRPaperReviewSample):
        sample.verify()
        self.samples.append(sample)

    def verify_all(self):
        for s in self.samples:
            s.verify()

register_dataset("iclr_review", ICLRReviewDataset)
