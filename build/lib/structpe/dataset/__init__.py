"""
dataset package: houses all dataset definitions plus a registry to manage them.
"""

# The registry file
from structpe.dataset.registry import DATASET_CLASSES, register_dataset, get_dataset_class

# Import each dataset file so that the register_dataset() calls happen
from structpe.dataset.search_dataset import SearchDataset
from structpe.dataset.sentiment_dataset import SentimentDataset
from structpe.dataset.iclr_review_dataset import ICLRReviewDataset
from structpe.dataset.conversation_dataset import ConversationDataset
from structpe.dataset.titanic_dataset import TitanicDataset
from structpe.dataset.hotel_booking_dataset import HotelBookingDataset

# If any new dataset is added, import it here:
# from structpe.dataset.my_new_dataset import MyNewDataset
