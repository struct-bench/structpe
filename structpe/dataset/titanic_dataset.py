import csv
import json
import os
from enum import Enum

from structpe.dataset.registry import register_dataset

class SexType(Enum):
    MALE = "male"
    FEMALE = "female"

class TitanicSample:
    """
    A single row from the Titanic dataset with minimal fields:
      - passenger_id
      - survived
      - pclass
      - name
      - sex
      - age
    Basic constraints:
      - survived in [0,1]
      - pclass in [1,2,3]
      - age >= 0
      - sex in {male, female}
    """

    def __init__(self, passenger_id, survived, pclass, name, sex, age):
        self.passenger_id = passenger_id
        self.survived = survived
        self.pclass = pclass
        self.name = name
        self.sex = sex
        self.age = age

    def check_sample_constraints(self, idx: int):
        # Check survived
        if self.survived not in [0,1]:
            print(f"[TitanicSample] (idx={idx}) WARNING: survived={self.survived}, expected 0 or 1.")
        # Check pclass
        if self.pclass not in [1,2,3]:
            print(f"[TitanicSample] (idx={idx}) WARNING: pclass={self.pclass}, expected 1,2,3.")
        # Check age
        if self.age < 0:
            print(f"[TitanicSample] (idx={idx}) WARNING: age={self.age}, expected >= 0.")
        # Check sex
        if self.sex not in [SexType.MALE, SexType.FEMALE]:
            print(f"[TitanicSample] (idx={idx}) WARNING: sex={self.sex}, expected male or female.")

class TitanicDataset:
    """
    Loads Titanic data from:
      - JSON: a list of dicts with keys: ["passenger_id","survived","pclass","name","sex","age"]
      - CSV/TSV: each row has the same columns in a header.

    No programmatic sample additions.
    """

    def __init__(self, file: str = None):
        self.samples = []
        if file:
            # decide how to load
            if file.endswith(".json"):
                self._load_from_json(file)
            elif file.endswith(".csv") or file.endswith(".tsv"):
                self._load_from_csv_tsv(file)
            else:
                raise ValueError(f"[TitanicDataset] Unsupported file format: {file}")

    def _load_from_json(self, filepath: str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"[TitanicDataset] JSON file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"[TitanicDataset] The JSON must be a list. Got {type(data)}")

        for idx, item in enumerate(data):
            passenger_id = item.get("passenger_id", None)
            survived = item.get("survived", 0)
            pclass = item.get("pclass", 3)
            name = item.get("name", "")
            sex_str = item.get("sex", "male")
            age_val = item.get("age", 0)

            # Convert sex
            sex = None
            sex_str_lower = str(sex_str).lower()
            if sex_str_lower in ["male","m"]:
                sex = SexType.MALE
            elif sex_str_lower in ["female","f"]:
                sex = SexType.FEMALE

            sample = TitanicSample(
                passenger_id=int(passenger_id) if passenger_id else -1,
                survived=int(survived),
                pclass=int(pclass),
                name=name,
                sex=sex,
                age=float(age_val)
            )
            sample.check_sample_constraints(idx)
            self.samples.append(sample)

        print(f"[TitanicDataset] Loaded {len(self.samples)} samples from '{filepath}'.")

    def _load_from_csv_tsv(self, filepath: str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"[TitanicDataset] CSV/TSV file not found: {filepath}")

        delimiter = "," if filepath.endswith(".csv") else "\t"

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for idx, row in enumerate(reader):
                passenger_id = row.get("passenger_id", None)
                survived = row.get("survived", "0")
                pclass = row.get("pclass", "3")
                name = row.get("name", "")
                sex_str = row.get("sex", "male")
                age_val = row.get("age", "0")

                # Convert sex
                sex = None
                sex_str_lower = str(sex_str).lower()
                if sex_str_lower in ["male","m"]:
                    sex = SexType.MALE
                elif sex_str_lower in ["female","f"]:
                    sex = SexType.FEMALE

                sample = TitanicSample(
                    passenger_id=int(passenger_id) if passenger_id else -1,
                    survived=int(survived),
                    pclass=int(pclass),
                    name=name,
                    sex=sex,
                    age=float(age_val)
                )
                sample.check_sample_constraints(idx)
                self.samples.append(sample)

        print(f"[TitanicDataset] Loaded {len(self.samples)} samples from '{filepath}'.")

    def verify_all(self):
        for idx, s in enumerate(self.samples):
            s.check_sample_constraints(idx)

# Register under "titanic"
register_dataset("titanic", TitanicDataset)
