from lark import Lark

GRAMMAR = r"""
    conversation: turn (turn)*
    turn: request response
    request: "USER: " ESCAPED_STRING
    response: "GPT: " ESCAPED_STRING

    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
"""

TEST_DATA = """
USER: "How are you?"

GPT: "I'm doing great, thanks for asking! How about you?"

USER "Good."

GPT: "Glad to hear that! What's on your mind today?"
"""

if __name__ == "__main__":
    try: 
        parser = Lark(GRAMMAR, start="conversation")
        print(parser.parse(TEST_DATA).pretty())
    except Exception as e:
        print("Error parsing input")