"""
product_name,overall_rating,rating,title,cleaned_review
"MILTON Vogue 750 Stainless Steel Water Bottle, Yellow 750 ml Bottle Reviews",4.1,5,Worth every penny,surprised see quality price thankx flipkart service thanks seller product
cello Puro Steel-X Royce 900 | PU Insulation | Leak Proof | Wide Mouth | 730 ml Bottle Reviews,4.1,5,Super!,nice
Virtuous 2000ml BPA Free Travel Leakproof Water Bottle for Home & Gym (Pack of 3) 2000 ml Bottle Reviews,4.0,5,Highly recommended,super
SPEEDEX Water for fridge School Gym Yoga Home office Boys Girls 1000 ml Bottle Reviews,4.0,5,Wonderful,great quality thanks flipkart z best bottle amount
pexpo 24 Hrs Hot and Cold Vacuum Insulated Water Bottle with Jute-bag Fererro 1000 ml Flask Reviews,4.3,5,Must buy!,quality bottle top notch comes cup awesome jute bag keep things hot cold longer time good product travel purpose regular use well purchased doubt
"MILTON Kool Trendy 500 Insulated Water with Straw for Kids, 490 ml Bottle Reviews",4.3,4,Value-for-money,nice bottle daughter happy
"BOROSIL Travelease Vacuum Insulate Bottle, 12 Hrs. Hot & 14 Hrs. Cold 420 ml Flask Reviews",4.2,5,Excellent,beautiful quality
MILTON Thermosteel Duo 750 DLX 700 ml Flask Reviews,4.4,5,Terrific purchase,happy product let see long product
MILTON Viva tuff 1000 ml Flask Reviews,4.1,5,Terrific purchase,great
"""

import csv
import json
import os
from enum import Enum

from structpe.dataset.registry import register_dataset

class WaterSample:
    def __init__(self, product_name, overall_rating, title, cleaned_review):
        self.product_name = product_name
        self.overall_rating = overall_rating
        self.title = title
        self.cleaned_review = cleaned_review
    def check_sample_constraints(self, idx: int):
        pass

class WaterDataset:
    def __init__(self, file: str = None):
        self.samples = []
        if file:
            if file.endswith(".csv"):
                with open(file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        sample = WaterSample(
                            str(row["product_name"]),
                            float(row["overall_rating"]) if row["overall_rating"] != '' else 0.0,
                            str(row["title"]),
                            str(row["cleaned_review"])
                        )
                        sample.check_sample_constraints(idx)
                        self.samples.append(sample)
    
    def verify_all(self):
        for idx, s in enumerate(self.samples):
            s.check_sample_constraints(idx)

register_dataset("water", WaterDataset)


# {"age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors", "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"}

WATER_GRAMMAR = r"""
start: "{" "'product_name'" ":" SINGLE_QUOTED_STRING "," "'overall_rating'" ":" FLOAT "," "'title'" ":" SINGLE_QUOTED_STRING "," "'cleaned_review'" ":" SINGLE_QUOTED_STRING "}"

SINGLE_QUOTED_STRING: /'([^'\\]*(\\.[^'\\]*)*)'/

%import common.INT
%import common.ESCAPED_STRING
%import common.WS
%import common.FLOAT
%ignore WS
"""

WATER_GRAMMAR = r"""
start: "{" "'product_name'" ":" (SINGLE_QUOTED_STRING | ESCAPED_STRING) "," "'overall_rating'" ":" FLOAT "," "'title'" ":" (SINGLE_QUOTED_STRING | ESCAPED_STRING) "," "'cleaned_review'" ":" (SINGLE_QUOTED_STRING | ESCAPED_STRING) "}"

SINGLE_QUOTED_STRING: /'([^'\\]*(\\.[^'\\]*)*)'/

%import common.INT
%import common.ESCAPED_STRING
%import common.WS
%import common.FLOAT
%ignore WS
"""

def get_grammar():
    """
    Return the grammar string so GenericGrammarCheck can call it if needed.
    """
    return WATER_GRAMMAR


compute_node_similarities = [
    ("title", "cleaned_review")
]

# class SexType(Enum):
#     MALE = "male"
#     FEMALE = "female"

# class TitanicSample:
#     """
#     A single row from the Titanic dataset with minimal fields:
#       - passenger_id
#       - survived
#       - pclass
#       - name
#       - sex
#       - age
#     Basic constraints:
#       - survived in [0,1]
#       - pclass in [1,2,3]
#       - age >= 0
#       - sex in {male, female}
#     """

#     def __init__(self, passenger_id, survived, pclass, name, sex, age):
#         self.passenger_id = passenger_id
#         self.survived = survived
#         self.pclass = pclass
#         self.name = name
#         self.sex = sex
#         self.age = age

#     def check_sample_constraints(self, idx: int):
#         # Check survived
#         if self.survived not in [0,1]:
#             print(f"[TitanicSample] (idx={idx}) WARNING: survived={self.survived}, expected 0 or 1.")
#         # Check pclass
#         if self.pclass not in [1,2,3]:
#             print(f"[TitanicSample] (idx={idx}) WARNING: pclass={self.pclass}, expected 1,2,3.")
#         # Check age
#         if self.age < 0:
#             print(f"[TitanicSample] (idx={idx}) WARNING: age={self.age}, expected >= 0.")
#         # Check sex
#         if self.sex not in [SexType.MALE, SexType.FEMALE]:
#             print(f"[TitanicSample] (idx={idx}) WARNING: sex={self.sex}, expected male or female.")

# class TitanicDataset:
#     """
#     Loads Titanic data from:
#       - JSON: a list of dicts with keys: ["passenger_id","survived","pclass","name","sex","age"]
#       - CSV/TSV: each row has the same columns in a header.

#     No programmatic sample additions.
#     """

#     def __init__(self, file: str = None):
#         self.samples = []
#         if file:
#             # decide how to load
#             if file.endswith(".json"):
#                 self._load_from_json(file)
#             elif file.endswith(".csv") or file.endswith(".tsv"):
#                 self._load_from_csv_tsv(file)
#             else:
#                 raise ValueError(f"[TitanicDataset] Unsupported file format: {file}")

#     def _load_from_json(self, filepath: str):
#         if not os.path.isfile(filepath):
#             raise FileNotFoundError(f"[TitanicDataset] JSON file not found: {filepath}")

#         with open(filepath, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         if not isinstance(data, list):
#             raise ValueError(f"[TitanicDataset] The JSON must be a list. Got {type(data)}")

#         for idx, item in enumerate(data):
#             passenger_id = item.get("passenger_id", None)
#             survived = item.get("survived", 0)
#             pclass = item.get("pclass", 3)
#             name = item.get("name", "")
#             sex_str = item.get("sex", "male")
#             age_val = item.get("age", 0)

#             # Convert sex
#             sex = None
#             sex_str_lower = str(sex_str).lower()
#             if sex_str_lower in ["male","m"]:
#                 sex = SexType.MALE
#             elif sex_str_lower in ["female","f"]:
#                 sex = SexType.FEMALE

#             sample = TitanicSample(
#                 passenger_id=int(passenger_id) if passenger_id else -1,
#                 survived=int(survived),
#                 pclass=int(pclass),
#                 name=name,
#                 sex=sex,
#                 age=float(age_val)
#             )
#             sample.check_sample_constraints(idx)
#             self.samples.append(sample)

#         print(f"[TitanicDataset] Loaded {len(self.samples)} samples from '{filepath}'.")

#     def _load_from_csv_tsv(self, filepath: str):
#         if not os.path.isfile(filepath):
#             raise FileNotFoundError(f"[TitanicDataset] CSV/TSV file not found: {filepath}")

#         delimiter = "," if filepath.endswith(".csv") else "\t"

#         with open(filepath, "r", encoding="utf-8") as f:
#             reader = csv.DictReader(f, delimiter=delimiter)
#             for idx, row in enumerate(reader):
#                 passenger_id = row.get("passenger_id", None)
#                 survived = row.get("survived", "0")
#                 pclass = row.get("pclass", "3")
#                 name = row.get("name", "")
#                 sex_str = row.get("sex", "male")
#                 age_val = row.get("age", "0")

#                 # Convert sex
#                 sex = None
#                 sex_str_lower = str(sex_str).lower()
#                 if sex_str_lower in ["male","m"]:
#                     sex = SexType.MALE
#                 elif sex_str_lower in ["female","f"]:
#                     sex = SexType.FEMALE

#                 sample = TitanicSample(
#                     passenger_id=int(passenger_id) if passenger_id else -1,
#                     survived=int(survived),
#                     pclass=int(pclass),
#                     name=name,
#                     sex=sex,
#                     age=float(age_val)
#                 )
#                 sample.check_sample_constraints(idx)
#                 self.samples.append(sample)

#         print(f"[TitanicDataset] Loaded {len(self.samples)} samples from '{filepath}'.")

#     def verify_all(self):
#         for idx, s in enumerate(self.samples):
#             s.check_sample_constraints(idx)

# # Register under "titanic"
# register_dataset("titanic", TitanicDataset)
