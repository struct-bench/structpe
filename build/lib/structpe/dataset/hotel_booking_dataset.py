"""
Hotel booking dataset with multiple constraints:
 - booking_id non-empty
 - check_in_date < check_out_date
 - num_guests >= 1
 - total_price >= 0
 - stay_length >= 1 day
"""

import datetime
from structpe._types import AtomicRangeInt, AtomicRangeFloat
from structpe.dataset.registry import register_dataset

class HotelBookingSample:
    """
    Represents a single hotel booking with constraints.
    """
    def __init__(
        self,
        booking_id: str,
        check_in_date: str,   # "YYYY-MM-DD"
        check_out_date: str,  # "YYYY-MM-DD"
        num_guests: AtomicRangeInt,  # must be >=1
        total_price: AtomicRangeFloat # must be >=0
    ):
        self.booking_id = booking_id
        self.check_in_date = check_in_date
        self.check_out_date = check_out_date
        self.num_guests = num_guests
        self.total_price = total_price
        self.verify()

    def verify(self):
        # booking_id cannot be empty
        if not self.booking_id.strip():
            raise ValueError("Booking ID cannot be empty.")

        # Parse check_in_date and check_out_date to date objects
        try:
            check_in_dt = datetime.datetime.strptime(self.check_in_date, "%Y-%m-%d").date()
            check_out_dt = datetime.datetime.strptime(self.check_out_date, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format.")

        if check_in_dt >= check_out_dt:
            raise ValueError(f"Check-in {self.check_in_date} must be before check-out {self.check_out_date}.")

        # num_guests and total_price are already validated by their range classes
        # But let's also ensure there's at least 1 day difference
        stay_length = (check_out_dt - check_in_dt).days
        if stay_length < 1:
            raise ValueError(f"Stay length must be >= 1 day. Computed={stay_length}.")

class HotelBookingDataset:
    def __init__(self):
        self.samples = []

    def add_sample(self, sample: HotelBookingSample):
        sample.verify()
        self.samples.append(sample)

    def verify_all(self):
        for sample in self.samples:
            sample.verify()

register_dataset("hotel_booking", HotelBookingDataset)
