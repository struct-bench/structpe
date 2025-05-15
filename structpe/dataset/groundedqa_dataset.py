import json
import os
from enum import Enum

from structpe.dataset.registry import register_dataset
from structpe.dataset.decorators import dataset_metric
from structpe.utilities.graph_handling import topological_sort

###############################################################################
# Enums (optional)
###############################################################################
# Define any enums if needed. Empty for now.
# class QARelevancy(Enum):
#     RELEVANT = "RELEVANT"
#     NOT_RELEVANT = "NOT_RELEVANT"

###############################################################################
# Sample Class
###############################################################################
class GroundedQASample:
    """
    Represents one QA sample with:
      source1, source2, query, and response.

    'sample_graph' (reversed adjacency) indicates 'response' depends on
    'source1', 'source2', and 'query'.
    
    'check_sample_constraints()' ensures each text field has at least 3 words.
    Also checks for cycles using topological_sort.
    """

    def __init__(self, source1, source2, query, response):
        self.source1 = source1
        self.source2 = source2
        self.query = query
        self.response = response

        # Reversed adjacency: "response" depends on the others
        self.sample_graph = {}

    def check_sample_constraints(self, idx: int):
        """
        - each text field >=3 words (if not None/empty)
        - no cycles in sample_graph
        """
        self._check_min_words(idx, "source1", self.source1)
        self._check_min_words(idx, "source2", self.source2)
        self._check_min_words(idx, "query",   self.query)
        self._check_min_words(idx, "response", self.response)

        self._check_sample_graph(idx)

    def _check_min_words(self, idx: int, field_name: str, text_val: str, min_words: int = 3):
        if text_val:
            wc = len(text_val.split())
            if wc < min_words:
                print(f"[GroundedQASample] (idx={idx}) WARNING: '{field_name}' has {wc} words (<{min_words}).")

    def _check_sample_graph(self, idx: int):
        try:
            order = topological_sort(self.sample_graph)
            print(f"[GroundedQASample] (idx={idx}) Graph generation order: {order}")
        except ValueError as e:
            print(f"[GroundedQASample] (idx={idx}) GRAPH ERROR: {e}")

###############################################################################
# Dataset Class
###############################################################################
class GroundedQADataset:
    """
    Loads from JSON, expecting each item with keys:
      'source1', 'source2', 'query', 'response'.

    Stored as GroundedQASample objects in self.samples.
    We auto-check constraints on each sample.
    """

    def __init__(self, file: str = None):
        self.samples = []
        if file:
            self._load_from_json(file)

    def _load_from_json(self, filepath: str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"[GroundedQADataset] JSON file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"[GroundedQADataset] The JSON must be a list. Got {type(data)}")

        for idx, item in enumerate(data):
            source1_val  = item.get("source1", None)
            source2_val  = item.get("source2", None)
            query_val    = item.get("query", None)
            response_val = item.get("response", None)

            # If everything is None => skip
            if (source1_val is None and source2_val is None
                    and query_val is None and response_val is None):
                raise ValueError(f"[GroundedQADataset] (idx={idx}) All fields are None. Cannot load sample.")

            sample = GroundedQASample(source1_val, source2_val, query_val, response_val)
            sample.check_sample_constraints(idx)
            self.samples.append(sample)

        print(f"[GroundedQADataset] Loaded {len(self.samples)} samples from '{filepath}'.")

    def verify_dataset(self):
        """
        Basic dataset-level checks.
        Returns a list of (index, pass_bool, reason).
        """
        results = []
        for i, s in enumerate(self.samples):
            # Example: mark all as 'Sample-level OK' for now
            results.append((i, True, "Sample-level OK."))
        return results
    
    def get_pe_random_api_prompt(self) -> str:
        """
        LLM prompt for random QA generation:
        'response' depends on 'source1','source2','query'.
        """
        return """Generate a JSON with four keys: source1, source2, query, response.
    'response' depends on the three other fields. Each field >=3 words.
    This is a QA item.
    'source1' and 'source2' are text sources.
    'query' is a question about the sources.
    'response' is the answer to the question.
    If the query cannot be answered by source1/source2, say "cannot be answered."
    """

    def get_pe_variation_api_prompt(self) -> str:
        """
        Similar prompt for rewriting existing data.
        """
        return """Rewrite this QA item, respecting adjacency:
    response => source1,source2,query.
    Ensure each text field >=3 words; if unanswerable, say "cannot be answered."
    """

###############################################################################
# Optional Prompts for Generation
###############################################################################


###############################################################################
# Optional Grammar
###############################################################################
GROUND_QA_GRAMMAR = r"""
start: source1_line newline source2_line newline query_line newline response_line newline?

source1_line: "source1" ":" ESCAPED_STRING
source2_line: "source2" ":" ESCAPED_STRING
query_line:   "query"   ":" ESCAPED_STRING
response_line:"response": ESCAPED_STRING

newline: /(\r?\n)+/

%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""

def get_grammar():
    return GROUND_QA_GRAMMAR

def build_grammar_string_for_check(sample):
    """
    Convert a GroundedQASample into a multiline string for grammar checking (if needed).
    """
    lines = []
    s1 = sample.source1 or ""
    s2 = sample.source2 or ""
    q  = sample.query or ""
    r  = sample.response or ""
    lines.append(f'source1:"{s1}"')
    lines.append(f'source2:"{s2}"')
    lines.append(f'query:"{q}"')
    lines.append(f'response:"{r}"')
    return "\n".join(lines)

###############################################################################
# Example Dataset-Level Metric
###############################################################################
@dataset_metric(level="dataset")
def dataset_metric_unanswerable_count(dataset_obj, all_samples_results):
    """
    Counts how many responses mention 'cannot be answered'.
    """
    count_unanswerable = 0
    for s in dataset_obj.samples:
        resp = (s.response or "").lower()
        if "cannot be answered" in resp:
            count_unanswerable += 1

    total = len(dataset_obj.samples)
    ratio = count_unanswerable / total if total else 0
    return {
        "unanswerable_count": count_unanswerable,
        "unanswerable_ratio": round(ratio, 4),
        "total_samples": total
    }

###############################################################################
# Register the Dataset
###############################################################################
register_dataset("groundedqa", GroundedQADataset)
