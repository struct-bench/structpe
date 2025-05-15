from lark import Lark

GRAMMARS = {
    "conversation": r"""
        conversation: turn (turn)*
        turn: request response
        request: "USER: " ESCAPED_STRING
        response: "GPT: " ESCAPED_STRING

        %import common.ESCAPED_STRING
        %import common.WS
        %ignore WS
    """,

    "math_expression": r"""
        start: expression
        expression: term (("+"|"-") term)*
        term: factor (("*"|"/") factor)*
        factor: NUMBER | "(" expression ")"

        %import common.NUMBER
        %import common.WS
        %ignore WS
    """,

    "date_format": r"""
        start: date
        date: /\d{4}-\d{2}-\d{2}/  // YYYY-MM-DD format

        %import common.WS
        %ignore WS
    """,

    "log_entry": r"""
        start: timestamp " - " level " - " message
        timestamp: /\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/
        level: "INFO" | "WARNING" | "ERROR"
        message: /.+/

        %import common.WS
        %ignore WS
    """,
}

# Build a parser for each grammar
PARSERS = {}
for name, grammar_text in GRAMMARS.items():
    # If a rule named 'start' exists, we use 'start'; else, we assume the entire grammar is the start
    start_rule = "start" if "start:" in grammar_text or "start :" in grammar_text else name
    PARSERS[name] = Lark(grammar_text, start=start_rule)
