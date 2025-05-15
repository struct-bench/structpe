"""
winner,conversation_a,conversation_b
model_b,"Question: Create me an itinerary for a 10 day trip to Jordan

Answer: Here is a possible itinerary for a 10-day trip to Jordan:
Day 1: Arrival in Amman
* Visit the Dead Sea and the Great Salt Lake * Visit the Temple of Moses, the ruins of the Temple Mount, and the Herodian Temple of the Demeter
* Explore the Old City, including the Citadel of the Kings, the Grand Bazaar, and the narrow alleys of the Old City
Day 2: Petra
* Visit the Valley of the Kings and the Temple Mount
* Visit the ruins of the Temple Mount, where the Israelites built the Temple Mount in the early 20th century
* Explore the Rosh Hashanah natural park, including the rock-cut tombs of the Prophet Muhammad
* Visit the Ash-Shabbat Valley, where the Israelites lived until the mid-20th century
Day 3: Petra
* Visit the Temple Mount and the tombs of the Prophet Muhammad
* Visit the Petra Museum, which features a collection of artifacts and exhibits from the Temple Mount excavations
* Explore the city of Petra, including the famous Hodjat al-Munawwara and the historic center of the city
Day 4: Petra
* Visit the Temple Mount and the tombs of the Prophet Muhammad
* Explore the city of Petra, including the Old City, the Temple Mount, and the site of the famous Zawiya mosque
* Visit the","Question: Create me an itinerary for a 10 day trip to Jordan

Answer: Day 1: Arrival in Amman and City Tour
- Check into hotel
- Visit the Roman Theater, the Temple of Hercules, and the Citadel
- Enjoy a traditional Jordanian dinner

Day 2: Petra Tour
- Take a day tour of Petra, including the Treasury, the Monastery, and the Siq
- Enjoy a traditional Jordanian lunch in Petra

Day 3: Wadi Rum Desert Tour
- Take a day tour of Wadi Rum, including a visit to the Lawrence Spring and the Lawrence Memorial
- Enjoy a traditional Jordanian lunch in a Bedouin tent

Day 4: Baptism Site and Mount Nebo Tour
- Visit the baptism site of Jesus on the Jordan River
- Visit Mount Nebo, where Moses saw the Promised Land
- Enjoy a traditional Jordanian dinner in Madaba

Day 5: Dead Sea Tour
- Visit the Dead Sea, the lowest point on earth
- Enjoy a traditional Jordanian lunch in a beachside restaurant

Day 6: Jerash and Ajloun Castle Tour
- Visit the Roman ruins of Jerash
- Visit the Ajloun Castle, a 12th century Crusader castle
- Enjoy a traditional Jordanian dinner in Ajloun

Day 7: Baptism Site and Dana Nature Reserve Tour
- Visit the baptism site of Jesus on the Jordan River
- Visit the Dana Nature Reserve, including the Dana Biosphere Reserve and the Dana Hotel
- Enjoy a traditional Jordanian dinner in Petra

Day 8: Qumran and Ein Gedi Tour
- Visit the Qumran caves where the Dead Sea Scrolls were found
- Visit Ein Gedi, a nature reserve with a beautiful waterfall
- Enjoy a traditional Jordanian lunch in Ein Gedi

Day 9: Jerash and Ajloun Castle Tour
- Visit the Roman ruins of Jerash
- Visit the Ajloun Castle, a 12th century Crusader castle
- Enjoy a traditional Jordanian dinner in Ajloun

Day 10: Departure from Amman
- Check out of hotel
- Departure from Amman to Queen Alia International Airport
- End of trip"
model_b,"Question: What is it to be human.

Answer: Being human means having a complex range of emotions, thoughts, and behaviors that make us unique individuals. It also means having the ability to reason, communicate, create, and learn from our experiences. As humans, we are capable of empathy, compassion, and understanding, which allows us to connect with others on a deep level. Additionally, being human also means having limitations and vulnerabilities that we must navigate and overcome throughout our lives.","Question: What is it to be human.

Answer: As an AI language model, I don't have personal beliefs or emotions, but here are some things that are often associated with the question of what it means to be human:

1. Being a conscious being with the ability to reason, feel, and make choices.
2. Having unique human characteristics such as language, culture, and social behavior.
3. Having the ability to create and appreciate art, music, literature, and other forms of expression.
4. Having the capacity for love, compassion, and empathy.
5. Being capable of experiencing pain and suffering.
6. Having the ability to learn, grow, and evolve throughout life.
7. Being accountable for one's actions and decisions.
8. Having a unique identity and sense of self.
9. Being capable of forming complex social relationships and societies.
10. Having the ability to make technological advancements and progress."
tie,"Question: I just got out of the Huixinxijienankou Station. Where do you think I am?

Answer: You are at Huixinxijienankou Station, which is a station on Line 5 and Line 10 of the Beijing Subway. It is located in the Xiaoguan district of Beijing, China. The station is close to the Huixin Xijie (New West Street) and Nankou (South Gate) areas. There are many shops and restaurants in the area, as well as a few hotels. The station is also close to the Beijing Zoo and the Olympic Green.

Here are some of the places you can go from Huixinxijienankou Station:

* The Beijing Zoo is about a 10-minute walk from the station.
* The Olympic Green is about a 20-minute walk from the station.
* The Huixin Xijie (New West Street) area is a popular shopping and dining area.
* The Nankou (South Gate) area is a historic area with many cultural attractions.
* There are a few hotels near the station, including the Huixin Hotel and the Nankou Hotel.

I hope this helps!","Question: I just got out of the Huixinxijienankou Station. Where do you think I am?

Answer: You are in Beijing, China. Huixinxijienankou Station is a station on Line 5 and Line 10 of the Beijing Subway."
model_b,"Question: Can stopping phenibut cause withdrawal symptoms? 

Answer: Yes, stopping phenibut abruptly or reducing the dosage too quickly can cause withdrawal symptoms. Phenibut is a central nervous system (CNS) depressant, and stopping its use can cause symptoms such as anxiety, irritability, insomnia, tremors, nausea, and seizures.","Question: Can stopping phenibut cause withdrawal symptoms? 

Answer: Yes, stopping phenibut can cause withdrawal symptoms.

Phenibut is a central nervous system depressant that can cause withdrawal symptoms when use is discontinued or reduced. Symptoms of phenibut withdrawal can include:

* Anxiety, agitation, and irritability
* Insomnia and difficulty sleeping
* Nausea and vomiting
* Headache
* Muscle pain and cramps
* Fatigue and lack of energy

It is important to taper off phenibut use under the supervision of a healthcare provider or a qualified healthcare professional to minimize the risk of withdrawal symptoms and ensure a safe and effective discontinuation of treatment."
"""

import csv
import json
import os
from enum import Enum

from structpe.dataset.registry import register_dataset

class ArenaSample:
    def __init__(self, conversation_a, conversation_b):
        self.conversation_a = conversation_a
        self.conversation_b = conversation_b
    def check_sample_constraints(self, idx: int):
        pass

class ArenaDataset:
    def __init__(self, file: str = None):
        self.samples = []
        if file:
            if file.endswith(".csv"):
                with open(file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        sample = ArenaSample(
                            str(row["conversation_a"]),
                            str(row["conversation_b"])
                        )
                        sample.check_sample_constraints(idx)
                        self.samples.append(sample)
    
    def verify_all(self):
        for idx, s in enumerate(self.samples):
            s.check_sample_constraints(idx)

register_dataset("arena", ArenaDataset)


# {"age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors", "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"}

ARENA_GRAMMAR = r"""
start: "{" "'conversation_a'" ":" (SINGLE_QUOTED_STRING | ESCAPED_STRING) "," "'conversation_b'" ":" (SINGLE_QUOTED_STRING | ESCAPED_STRING) "}"

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
    return ARENA_GRAMMAR


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
