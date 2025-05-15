from lark import Lark

json_grammar = r"""
start: "{" "'product_name'" ":" (SINGLE_QUOTED_STRING | ESCAPED_STRING) "," "'overall_rating'" ":" FLOAT "," "'title'" ":" (SINGLE_QUOTED_STRING | ESCAPED_STRING) "," "'cleaned_review'" ":" (SINGLE_QUOTED_STRING | ESCAPED_STRING) "}"

SINGLE_QUOTED_STRING: /'([^'\\]*(\\.[^'\\]*)*)'/

%import common.INT
%import common.ESCAPED_STRING
%import common.WS
%import common.FLOAT
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
    json_text = "{'product_name': 'Virtuous 2000ml BPA Free Travel Leakproof Water Bottle for Home & Gym (Pack of 3) 2000 ml Bottle Reviews', 'overall_rating': 4.0, 'title': 'Highly recommended', 'cleaned_review': 'super'}"
    json_text = "{'product_name': 'Apeiron Stainless Steel Fridge Water Bottle Silver (Pack of 6) 1000 ml Bottle Reviews', 'overall_rating': 3.8, 'title': \"Don't waste your money\", 'cleaned_review': 'leakage problem face'}"
    tree = parser.parse(json_text)
    print(tree.pretty())