from lark import Lark

json_grammar = r"""
    ?start: value

    ?value: object
          | array
          | string
          | SIGNED_NUMBER      -> number
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null

    object: "{" [pair ("," pair)*] "}"
    pair: string ":" value

    array: "[" [value ("," value)*] "]"

    string: ESCAPED_STRING

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""

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
    tree = parser.parse(json_text)
    print(tree.pretty())