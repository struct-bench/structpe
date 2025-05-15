from lark import Lark

json_grammar = r"""
start: "{" "'age'" ":" INT "," "'workclass'" ":" workclass "," "'fnlwgt'" ":" INT "," "'education'" ":" education "," "'education_num'" ":" INT "," "'marital_status'" ":" marital_status "," "'occupation'" ":" occupation "," "'relationship'" ":" relationship "," "'race'" ":" race "," "'sex'" ":" sex "," "'capital_gain'" ":" INT "," "'capital_loss'" ":" INT "," "'hours_per_week'" ":" INT "," "'native_country'" ":" native_country "}"

workclass: "'State-gov'" | "'Self-emp-not-inc'" | "'Private'" | "'Federal-gov'" | "'Local-gov'" | "'Self-emp-inc'" | "'Without-pay'" | "'Never-worked'"

education: "'Bachelors'" | "'Some-college'" | "'11th'" | "'HS-grad'" | "'Prof-school'" | "'Assoc-acdm'" | "'Assoc-voc'" | "'9th'" | "'7th-8th'" | "'12th'" | "'Masters'" | "'1st-4th'" | "'10th'" | "'Doctorate'" | "'5th-6th'" | "'Preschool'"

marital_status: "'Married-civ-spouse'" | "'Divorced'" | "'Never-married'" | "'Separated'" | "'Widowed'" | "'Married-spouse-absent'" | "'Married-AF-spouse'"

occupation: "'Tech-support'" | "'Craft-repair'" | "'Other-service'" | "'Sales'" | "'Exec-managerial'" | "'Prof-specialty'" | "'Handlers-cleaners'" | "'Machine-op-inspct'" | "'Adm-clerical'" | "'Farming-fishing'" | "'Transport-moving'" | "'Priv-house-serv'" | "'Protective-serv'" | "'Armed-Forces'"

relationship: "'Wife'" | "'Own-child'" | "'Husband'" | "'Not-in-family'" | "'Other-relative'" | "'Unmarried'"

race: "'White'" | "'Asian-Pac-Islander'" | "'Amer-Indian-Eskimo'" | "'Other'" | "'Black'"

sex: "'Female'" | "'Male'"

native_country: "'United-States'" | "'Cambodia'" | "'England'" | "'Puerto-Rico'" | "'Canada'" | "'Germany'" | "'Outlying-US(Guam-USVI-etc)'" | "'India'" | "'Japan'" | "'Greece'" | "'South'" | "'China'" | "'Cuba'" | "'Iran'" | "'Honduras'" | "'Philippines'" | "'Italy'" | "'Poland'" | "'Jamaica'" | "'Vietnam'" | "'Mexico'" | "'Portugal'" | "'Ireland'" | "'France'" | "'Dominican-Republic'" | "'Laos'" | "'Ecuador'" | "'Taiwan'" | "'Haiti'" | "'Columbia'" | "'Hungary'" | "'Guatemala'" | "'Nicaragua'" | "'Scotland'" | "'Thailand'" | "'Yugoslavia'" | "'El-Salvador'" | "'Trinadad&Tobago'" | "'Peru'" | "'Hong'" | "'Holand-Netherlands'"

%import common.INT
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""

# start: text_line newline sentiment_line (newline emotion_line)? (newline rating_line)? newline?

# text_line: "text" ":" ESCAPED_STRING
# sentiment_line: "sentiment" ":" SENTIMENT_VAL
# emotion_line: "emotion" ":" EMOTION_VAL
# rating_line: "rating" ":" RATING_VAL

# SENTIMENT_VAL: "POSITIVE" | "NEGATIVE" | "NEUTRAL"
# EMOTION_VAL: "HAPPY" | "SAD" | "ANGRY" | "NEUTRAL"
# RATING_VAL: /[0-5]/

# newline: /(\r?\n)+/

# %import common.ESCAPED_STRING
# %import common.WS
# %ignore WS
# """

parser = Lark(json_grammar, start="start", parser="lalr")

# Example usage:
if __name__ == "__main__":
    json_text = """
    {"age": 39,
     "workclass": "State-gov",
     "fnlwgt": 77516,
     "education": "Bachelors",
     "education-num": 13,
     "marital-status": "Never-married",
     "occupation": "Adm-clerical",
     "relationship": "Not-in-family",
     "race": "White",
     "sex": "Male",
     "capital-gain": 2174,
     "capital-loss": 0,
     "hours-per-week": 40,
     "native-country": "United-States"}
    """
    json_text = "{'age': 28, 'workclass': '?', 'fnlwgt': 60726, 'education': 'HS-grad', 'education_num': 9, 'marital_status': 'Never-married', 'occupation': '?', 'relationship': 'Own-child', 'race': 'Black', 'sex': 'Male', 'capital_gain': 0, 'capital_loss': 0, 'hours_per_week': 40, 'native_country': 'United-States'}"
    tree = parser.parse(json_text)
    print(tree.pretty())