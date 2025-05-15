"""
test_dataset.py: Unit tests for verifying each dataset's functionality.
"""

import unittest

# For search
from structpe.dataset.search_dataset import SearchDataset, SearchSample
from structpe._types import QueryIntent, QueryTopic, AtomicRangeInt

# For sentiment
from structpe.dataset.sentiment_dataset import SentimentDataset, SentimentSample
from structpe._types import SentimentLabel

# For ICLR reviews
from structpe.dataset.iclr_review_dataset import (
    ICLRReviewDataset, ICLRPaperReviewSample, ICLRComment,
    ICLRCommentRole, ICLRRecommendation
)
from structpe._types import AtomicRangeFloat

# For conversation
from structpe.dataset.conversation_dataset import (
    ConversationDataset, ConversationSample, ConversationTurn, ConversationRole
)

# For Titanic
from structpe.dataset.titanic_dataset import (
    TitanicDataset, TitanicSample, SexEnum
)

# For hotel bookings
from structpe.dataset.hotel_booking_dataset import (
    HotelBookingDataset, HotelBookingSample
)

class TestSearchDataset(unittest.TestCase):
    def test_search_dataset(self):
        ds = SearchDataset()
        sample = SearchSample(
            query_text="hello world",
            intent=QueryIntent.NAVIGATIONAL,
            topic=QueryTopic.TRAVEL,
            word_count=AtomicRangeInt(2,1,10)
        )
        ds.add_sample(sample)
        self.assertEqual(len(ds.samples), 1)
        ds.verify_all()

class TestSentimentDataset(unittest.TestCase):
    def test_sentiment_dataset(self):
        ds = SentimentDataset()
        sample = SentimentSample("I love it", SentimentLabel.POSITIVE)
        ds.add_sample(sample)
        self.assertEqual(len(ds.samples), 1)
        ds.verify_all()

class TestICLRReviewDataset(unittest.TestCase):
    def test_iclr_review_dataset(self):
        ds = ICLRReviewDataset()
        sample = ICLRPaperReviewSample(
            paper_title="Test Paper",
            rating=AtomicRangeInt(8, 1, 10),
            confidence=AtomicRangeFloat(3.5, 1.0, 5.0),
            recommendation=ICLRRecommendation.WEAK_ACCEPT,
            comments=[ICLRComment("Looks good!", ICLRCommentRole.REVIEWER)]
        )
        ds.add_sample(sample)
        self.assertEqual(len(ds.samples), 1)
        ds.verify_all()

class TestConversationDataset(unittest.TestCase):
    def test_conversation_dataset(self):
        ds = ConversationDataset()
        sample = ConversationSample([
            ConversationTurn(ConversationRole.USER, "Hi, I'd like to order pizza."),
            ConversationTurn(ConversationRole.ASSISTANT, "Sure, which toppings?")
        ])
        ds.add_sample(sample)
        self.assertEqual(len(ds.samples), 1)
        ds.verify_all()

class TestTitanicDataset(unittest.TestCase):
    def test_titanic_dataset(self):
        ds = TitanicDataset()
        from structpe._types import AtomicRangeInt, AtomicRangeFloat
        sample = TitanicSample(
            name="John Smith",
            age=AtomicRangeInt(30, 0, 120),
            sex=SexEnum.MALE,
            fare=AtomicRangeFloat(10.5, 0.0, 600.0),
            survived=True
        )
        ds.add_sample(sample)
        self.assertEqual(len(ds.samples), 1)
        ds.verify_all()

class TestHotelBookingDataset(unittest.TestCase):
    def test_hotel_booking_dataset(self):
        ds = HotelBookingDataset()
        from structpe._types import AtomicRangeInt, AtomicRangeFloat
        sample = HotelBookingSample(
            booking_id="BKG-XYZ",
            check_in_date="2025-05-10",
            check_out_date="2025-05-12",
            num_guests=AtomicRangeInt(2, 1, 10),
            total_price=AtomicRangeFloat(300.0, 0.0, 5000.0)
        )
        ds.add_sample(sample)
        self.assertEqual(len(ds.samples), 1)
        ds.verify_all()

if __name__ == "__main__":
    unittest.main()
