"""
Titanic dataset with name, age, sex, fare, survived flags.
"""

from enum import Enum
from structpe._types import AtomicRangeInt, AtomicRangeFloat
from structpe.dataset.registry import register_dataset

class SexEnum(Enum):
    MALE = "male"
    FEMALE = "female"

class TitanicSample:
    """
    A single row (passenger) from Titanic data:
      - name (str)
      - age (AtomicRangeInt)
      - sex (SexEnum)
      - fare (AtomicRangeFloat)
      - survived (bool)
    """
    def __init__(
        self,
        name: str,
        age: AtomicRangeInt,
        sex: SexEnum,
        fare: AtomicRangeFloat,
        survived: bool
    ):
        self.name = name
        self.age = age
        self.sex = sex
        self.fare = fare
        self.survived = survived
        self.verify()

    def verify(self):
        if not self.name.strip():
            raise ValueError("TitanicSample: name is empty.")

class TitanicDataset:
    def __init__(self):
        self.samples = []

    def add_sample(self, sample: TitanicSample):
        sample.verify()
        self.samples.append(sample)

    def verify_all(self):
        for s in self.samples:
            s.verify()

register_dataset("titanic_dataset", TitanicDataset)
