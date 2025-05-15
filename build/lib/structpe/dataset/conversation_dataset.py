"""
A multi-turn conversation dataset, with a list of ConversationTurn objects for each sample.
"""

from enum import Enum
from structpe.dataset.registry import register_dataset

class ConversationRole(Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"

class ConversationTurn:
    """
    One turn in a conversation. 
    role: e.g. USER, ASSISTANT, SYSTEM
    text: The message content
    """
    def __init__(self, role: ConversationRole, text: str):
        self.role = role
        self.text = text
        self.verify()

    def verify(self):
        if not self.text.strip():
            raise ValueError("ConversationTurn text is empty.")

class ConversationSample:
    """
    A single multi-turn conversation, storing a list of ConversationTurn objects.
    """
    def __init__(self, turns: list[ConversationTurn]):
        self.turns = turns
        self.verify()

    def verify(self):
        if not self.turns:
            raise ValueError("ConversationSample must have at least one turn.")
        for turn in self.turns:
            turn.verify()

class ConversationDataset:
    def __init__(self):
        self.samples = []

    def add_sample(self, sample: ConversationSample):
        sample.verify()
        self.samples.append(sample)

    def verify_all(self):
        for s in self.samples:
            s.verify()

register_dataset("conversation_dataset", ConversationDataset)
