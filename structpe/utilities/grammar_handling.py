import sys
from lark import Lark, exceptions as lark_exceptions

def build_parser(grammar_text: str):
    """
    Build and return a Lark parser from a given grammar string.
    """
    return Lark(grammar_text, start="start")

def check_sample_against_grammar(parser, sample_string: str) -> bool:
    """
    Attempts to parse 'sample_string' using the provided 'parser'.
    Returns True if parse succeeds, else False.
    """
    try:
        parser.parse(sample_string)
        return True
    except lark_exceptions.LarkError:
        return False
