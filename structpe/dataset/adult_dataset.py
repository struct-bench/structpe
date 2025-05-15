"""age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income
39,State-gov,77516,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States,<=50K
50,Self-emp-not-inc,83311,Bachelors,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,13,United-States,<=50K
38,Private,215646,HS-grad,9,Divorced,Handlers-cleaners,Not-in-family,White,Male,0,0,40,United-States,<=50K
53,Private,234721,11th,7,Married-civ-spouse,Handlers-cleaners,Husband,Black,Male,0,0,40,United-States,<=50K
28,Private,338409,Bachelors,13,Married-civ-spouse,Prof-specialty,Wife,Black,Female,0,0,40,Cuba,<=50K
37,Private,284582,Masters,14,Married-civ-spouse,Exec-managerial,Wife,White,Female,0,0,40,United-States,<=50K
49,Private,160187,9th,5,Married-spouse-absent,Other-service,Not-in-family,Black,Female,0,0,16,Jamaica,<=50K
52,Self-emp-not-inc,209642,HS-grad,9,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,45,United-States,>50K
31,Private,45781,Masters,14,Never-married,Prof-specialty,Not-in-family,White,Female,14084,0,50,United-States,>50K
42,Private,159449,Bachelors,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5178,0,40,United-States,>50K
37,Private,280464,Some-college,10,Married-civ-spouse,Exec-managerial,Husband,Black,Male,0,0,80,United-States,>50K
30,State-gov,141297,Bachelors,13,Married-civ-spouse,Prof-specialty,Husband,Asian-Pac-Islander,Male,0,0,40,India,>50K
23,Private,122272,Bachelors,13,Never-married,Adm-clerical,Own-child,White,Female,0,0,30,United-States,<=50K
32,Private,205019,Assoc-acdm,12,Never-married,Sales,Not-in-family,Black,Male,0,0,50,United-States,<=50K
40,Private,121772,Assoc-voc,11,Married-civ-spouse,Craft-repair,Husband,Asian-Pac-Islander,Male,0,0,40,?,>50K
34,Private,245487,7th-8th,4,Married-civ-spouse,Transport-moving,Husband,Amer-Indian-Eskimo,Male,0,0,45,Mexico,<=50K"""

import csv
import json
import os
from enum import Enum

from structpe.dataset.registry import register_dataset

class AdultSample:
    def __init__(self, age, workclass, fnlwgt, education, education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country):
        self.age = age
        self.workclass = workclass
        self.fnlwgt = fnlwgt
        self.education = education
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country
    def check_sample_constraints(self, idx: int):
        pass

class AdultDataset:
    def __init__(self, file: str = None):
        self.samples = []
        if file:
            if file.endswith(".csv"):
                with open(file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        sample = AdultSample(
                            int(row["age"]),
                            row["workclass"],
                            int(row["fnlwgt"]),
                            row["education"],
                            int(row["education-num"]),
                            row["marital-status"],
                            row["occupation"],
                            row["relationship"],
                            row["race"],
                            row["sex"],
                            int(row["capital-gain"]),
                            int(row["capital-loss"]),
                            int(row["hours-per-week"]),
                            row["native-country"]
                        )
                        sample.check_sample_constraints(idx)
                        self.samples.append(sample)
    
    def verify_all(self):
        for idx, s in enumerate(self.samples):
            s.check_sample_constraints(idx)

register_dataset("adult", AdultDataset)


# {"age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors", "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"}

ADULT_GRAMMAR = r"""
start: "{" "'age'" ":" INT "," "'workclass'" ":" workclass "," "'fnlwgt'" ":" INT "," "'education'" ":" education "," "'education_num'" ":" INT "," "'marital_status'" ":" marital_status "," "'occupation'" ":" occupation "," "'relationship'" ":" relationship "," "'race'" ":" race "," "'sex'" ":" sex "," "'capital_gain'" ":" INT "," "'capital_loss'" ":" INT "," "'hours_per_week'" ":" INT "," "'native_country'" ":" native_country "}"

workclass: "'State-gov'" | "'Self-emp-not-inc'" | "'Private'" | "'Federal-gov'" | "'Local-gov'" | "'Self-emp-inc'" | "'Without-pay'" | "'Never-worked'" | "'?'"

education: "'Bachelors'" | "'Some-college'" | "'11th'" | "'HS-grad'" | "'Prof-school'" | "'Assoc-acdm'" | "'Assoc-voc'" | "'9th'" | "'7th-8th'" | "'12th'" | "'Masters'" | "'1st-4th'" | "'10th'" | "'Doctorate'" | "'5th-6th'" | "'Preschool'"

marital_status: "'Married-civ-spouse'" | "'Divorced'" | "'Never-married'" | "'Separated'" | "'Widowed'" | "'Married-spouse-absent'" | "'Married-AF-spouse'"

occupation: "'Tech-support'" | "'Craft-repair'" | "'Other-service'" | "'Sales'" | "'Exec-managerial'" | "'Prof-specialty'" | "'Handlers-cleaners'" | "'Machine-op-inspct'" | "'Adm-clerical'" | "'Farming-fishing'" | "'Transport-moving'" | "'Priv-house-serv'" | "'Protective-serv'" | "'Armed-Forces'" | "'?'"

relationship: "'Wife'" | "'Own-child'" | "'Husband'" | "'Not-in-family'" | "'Other-relative'" | "'Unmarried'"

race: "'White'" | "'Asian-Pac-Islander'" | "'Amer-Indian-Eskimo'" | "'Other'" | "'Black'"

sex: "'Female'" | "'Male'"

native_country: "'United-States'" | "'Cambodia'" | "'England'" | "'Puerto-Rico'" | "'Canada'" | "'Germany'" | "'Outlying-US(Guam-USVI-etc)'" | "'India'" | "'Japan'" | "'Greece'" | "'South'" | "'China'" | "'Cuba'" | "'Iran'" | "'Honduras'" | "'Philippines'" | "'Italy'" | "'Poland'" | "'Jamaica'" | "'Vietnam'" | "'Mexico'" | "'Portugal'" | "'Ireland'" | "'France'" | "'Dominican-Republic'" | "'Laos'" | "'Ecuador'" | "'Taiwan'" | "'Haiti'" | "'Columbia'" | "'Hungary'" | "'Guatemala'" | "'Nicaragua'" | "'Scotland'" | "'Thailand'" | "'Yugoslavia'" | "'El-Salvador'" | "'Trinadad&Tobago'" | "'Peru'" | "'Hong'" | "'Holand-Netherlands'" | "'?'"

%import common.INT
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""

def get_grammar():
    """
    Return the grammar string so GenericGrammarCheck can call it if needed.
    """
    return ADULT_GRAMMAR


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
