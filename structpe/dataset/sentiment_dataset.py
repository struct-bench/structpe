import json
import os
from enum import Enum

from structpe.dataset.registry import register_dataset
from structpe.dataset.decorators import dataset_metric
from structpe.utilities.graph_handling import topological_sort

class SentimentLabel(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

class EmotionType(Enum):
    HAPPY = "HAPPY"
    SAD = "SAD"
    ANGRY = "ANGRY"
    NEUTRAL = "NEUTRAL"

class SentimentSample:
    """
    All attributes in a sample will be used for evaluation.
    Represents one sample with text + sentiment + emotion + rating.
    Also includes a 'sample_graph' dict describing adjacency:
      sentiment => emotion, rating => text

    The 'check_sample_constraints()' method enforces:
     - text has >=3 words, if text not None
     - rating in [0..5], if rating not None
     - topological_sort for sample_graph (no cycles)
    """

    def __init__(self, text, sentiment, emotion, rating):
        self.text = text
        self.sentiment = sentiment
        self.emotion = emotion
        self.rating = rating

        # Reversed adjacency
        self.sample_graph = {}

    def check_sample_constraints(self, idx: int):
        """
        Basic constraints + adjacency check:
          - text >=3 words if not None
          - rating in [0..5] if not None
          - topological sort on sample_graph to ensure no cycles
        """
        # Check text length
        if self.text is not None:
            wc = len(self.text.split())
            if wc < 3:
                print(f"[SentimentSample] (idx={idx}) WARNING: text has {wc} words (<3).")

        # Check rating bounds
        if self.rating is not None:
            if not (0 <= self.rating <= 5):
                print(f"[SentimentSample] (idx={idx}) WARNING: rating {self.rating} not in [0..5].")

        # Finally check adjacency
        self.check_sample_graph(idx)

    def check_sample_graph(self, idx: int):
        """
        Attempt a topological sort on sample_graph. Logs if a cycle is found.
        """
        try:
            order = topological_sort(self.sample_graph)
            print(f"[SentimentSample] (idx={idx}) Graph generation order: {order}")
        except ValueError as e:
            print(f"[SentimentSample] (idx={idx}) GRAPH ERROR: {e}")

class SentimentDataset:
    """
    Loads from a JSON file. Each item is expected to have keys:
      "text", "sentiment", "emotion", "rating".

    We'll store them in self.samples as 'SentimentSample' objects.
    If an item has all None => skip with an error.
    We automatically call each sample's check_sample_constraints.
    """

    def __init__(self, file: str = None):
        self.samples = []
        if file:
            self._load_from_json(file)

    def _load_from_json(self, filepath: str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"[SentimentDataset] JSON file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"[SentimentDataset] The JSON must be a list. Got {type(data)}")

        for idx, item in enumerate(data):
            text = item.get("text", None)
            sentiment_str = item.get("sentiment", None)
            emotion_str = item.get("emotion", None)
            rating_val = item.get("rating", None)

            # If everything is None => skip with an error
            if text is None and sentiment_str is None and emotion_str is None and rating_val is None:
                raise ValueError(f"[SentimentDataset] (idx={idx}) All fields are None. Cannot load sample.")

            # Convert sentiment from string to enum
            sentiment = None
            if sentiment_str is not None:
                try:
                    sentiment = sentiment_str #SentimentLabel(sentiment_str.upper())
                except:
                    print(f"[SentimentDataset] (idx={idx}) WARNING: unrecognized sentiment '{sentiment_str}'. Using None")

            # Convert emotion from string to enum
            emotion = None
            if emotion_str is not None:
                try:
                    emotion = emotion_str #EmotionType(emotion_str.upper())
                except:
                    print(f"[SentimentDataset] (idx={idx}) WARNING: unrecognized emotion '{emotion_str}'. Using None")

            # Convert rating to int if possible
            if rating_val is not None and not isinstance(rating_val, int):
                print(f"[SentimentDataset] (idx={idx}) WARNING: rating not int => {rating_val}, using None")
                rating_val = None

            # Build the sample
            sample = SentimentSample(text, sentiment, emotion, rating_val)
            #sample.check_sample_constraints(idx)
            self.samples.append(sample)

        print(f"[SentimentDataset] Loaded {len(self.samples)} samples from '{filepath}'.")

    def verify_dataset(self):
        """
        We can do dataset-level checks here. e.g. no single sentiment is >80%.
        Returns list of (index, bool, reason).
        """
        from collections import Counter
        results = []

        # Sample-level checks
        for i, s in enumerate(self.samples):
            ok = True
            reason = "Sample-level OK"
            if s.text:
                wc = len(s.text.split())
                if wc < 3:
                    ok = False
                    reason = f"Text too short ({wc} words)."

            if s.rating is not None:
                if not (0 <= s.rating <= 5):
                    ok = False
                    reason = f"Rating {s.rating} out of [0..5]."

            results.append((i, ok, reason))

        # dataset-level: no single sentiment >80%
        real_sents = [s.sentiment for s in self.samples if s.sentiment is not None]
        total_s = len(real_sents)
        if total_s > 0:
            c = Counter(real_sents)
            for senti, count in c.items():
                ratio = count / total_s
                if ratio > 0.8:
                    results.append((-1, False, f"Dataset-level FAIL: {senti.value} is {ratio*100:.1f}% >80%."))
                    break

        # rating average ignoring None
        rated = [s for s in self.samples if s.rating is not None]
        if rated:
            avg = sum(s.rating for s in rated)/len(rated)
            for s in rated:
                if s.rating > avg + 3:
                    results.append((-1, False, f"Dataset-level FAIL: rating {s.rating} >> avg {avg:.2f}+3."))
                    break

        return results


def get_pe_random_api_prompt() -> str:
    return """You are an AI generating short text with these attributes in JSON:
sentiment => emotion + rating => text.
Each attribute is a node in reversed adjacency:
 - sentiment: []
 - emotion: [sentiment]
 - rating: [sentiment]
 - text: [sentiment,emotion]
Ensure text >=3 words, rating in [0..5].
"""

def get_pe_variation_api_prompt() -> str:
    return """Rewrite this JSON while respecting the reversed adjacency:
sentiment => emotion,rating => text.
No cycles, text >=3 words, rating in [0..5].
"""

# The grammar used by the GenericGrammarCheck
SENTIMENT_GRAMMAR = r"""
start: "{" "'text'" ":" SINGLE_QUOTED_STRING "," "'sentiment'" ":" SINGLE_QUOTED_STRING "," "'emotion'" ":" SINGLE_QUOTED_STRING "," "'rating'" ":" INT "}"

SINGLE_QUOTED_STRING: /'([^'\\]*(\\.[^'\\]*)*)'/

%import common.INT
%import common.WS
%ignore WS
"""

def get_grammar():
    """
    Return the grammar string so GenericGrammarCheck can call it if needed.
    """
    return SENTIMENT_GRAMMAR

# def build_grammar_string_for_check(sample):
#     """
#     Convert a SentimentSample into a single-line string to parse with the grammar,
#     safely escaping any " or \ inside the text so it won't break the ESCAPED_STRING rule.
#     """
#     parts = []

#     # text
#     txt = getattr(sample, "text", "")
#     if not txt:
#         txt = ""
#     else:
#         # Escape any backslashes and double quotes so Lark sees valid "..." content
#         txt = txt.replace("\\", "\\\\").replace('"', '\\"')
#     parts.append(f'text:"{txt}"')

#     # sentiment
#     sentiment_obj = getattr(sample, "sentiment", None)
#     if sentiment_obj:
#         parts.append(f'sentiment:{sentiment_obj.value}')
#     else:
#         parts.append('sentiment:NEUTRAL')

#     # emotion
#     emotion_obj = getattr(sample, "emotion", None)
#     if emotion_obj:
#         parts.append(f'emotion:{emotion_obj.value}')

#     # rating
#     rating_val = getattr(sample, "rating", None)
#     if rating_val is not None:
#         parts.append(f'rating:{rating_val}')

#     # Join into one line, e.g.:
#     # text:"Incredible reliability" sentiment:POSITIVE emotion:HAPPY rating:5
#     return ' '.join(parts)

@dataset_metric(level="dataset")
def dataset_metric_sentiment_accuracy(dataset_obj, all_samples_results):
    """
    Example: a dataset-level metric that checks overall sentiment correctness.
    """
    correct_count = 0
    total = 0
    for idx, sample in enumerate(dataset_obj.samples):
        rating = sample.rating
        sentiment = sample.sentiment.value if sample.sentiment else "NEUTRAL"
        if rating is not None:
            total += 1
            if rating > 2 and sentiment == "POSITIVE":
                correct_count += 1
            elif rating <= 2 and sentiment in ["NEGATIVE", "NEUTRAL"]:
                correct_count += 1
    accuracy = correct_count / total if total else 0.0
    return {
        "sentiment_accuracy": round(accuracy, 4),
        "correct_count": correct_count,
        "total_samples": total
    }

@dataset_metric(level="dataset")
def dataset_metric_rating_distribution(dataset_obj, all_samples_results):
    """
    Another dataset-level metric returning rating distribution across the dataset.
    """
    from collections import Counter
    rating_vals = [s.rating for s in dataset_obj.samples if s.rating is not None]
    dist = Counter(rating_vals)
    dist_dict = {r: c for r, c in dist.items()}
    return {
        "rating_distribution": dist_dict,
        "num_rated_samples": len(rating_vals)
    }

@dataset_metric(level="sample")
def dataset_metric_sentiment_sample_grade(dataset_obj, single_sample_result):
    """
    Example: a sample-level metric that you might want to compute for each sample.
    We'll produce a 'grade' for each sample, based on rating>2 => 'A' else 'B'.
    """
    sample = single_sample_result["raw_sample_obj"]
    rating = sample.rating
    if rating is None:
        return {"grade": "NONE"}
    return {"grade": "A" if rating > 2 else "B"}

# Some node pairs for grammar_check aggregator
compute_node_similarities = [
    ("text", "sentiment"),
    ("sentiment", "emotion"),
    ("emotion", "rating")
]

# ---------------------
# TREGEX QUERIES
# ---------------------
# We assume the parse nodes from SENTIMENT_GRAMMAR are:
#   start, text_line, sentiment_line, emotion_line, rating_line, newline

tregex_queries = [
    # 1) Capture the root node 'start'
    "start",

    # 2) Capture the 'text_line' node
    "text_line=txt",

    # 3) Capture the 'sentiment_line' node
    "sentiment_line=sent",

    # 4) Capture the 'emotion_line' node
    "emotion_line=emo",

    # 5) Capture the 'rating_line' node
    "rating_line=rat",

    # 6) Parent-child: 'start' has a child 'text_line'
    "start < text_line",

    # 7) Parent-child: 'start' has a child 'sentiment_line'
    "start < sentiment_line",

    # 8) Check that 'start' is NOT child-of 'something'
    "!> start",

    # 9) 'start' that has a child 'rating_line'
    "start < rating_line",

    # 10) Negation: 'start' not containing 'emotion_line'
    "start !< emotion_line",

    # 11) rating_line=rate that has sibling sentiment_line=sent
    "(rating_line=rate $ sentiment_line=sent)",

    # 12) If emotion_line is present, capture it
    "emotion_line=any_emo",

    # 13) 'start' node that has child 'newline'
    "start < newline",

    # 14) Or-pattern: either 'text_line' or 'sentiment_line'
    "(text_line | sentiment_line)",

    # 15) Conjunction: match a node that is both 'rating_line'
    #    AND has sibling 'emotion_line' (all in the same parent).
    "(rating_line & (rating_line $ emotion_line))",

    # -----------------------------
    # EXTRA Content-Specific QUERIES
    # -----------------------------

    # 16) sentiment_line that contains 'POSITIVE' in its leaves
    #     (We use < /POSITIVE/ to match leaves with regex "POSITIVE")
    "(sentiment_line < /POSITIVE/)",

    # 17) rating_line that specifically has '5' among its leaves
    "(rating_line < /5/)",

    # 18) text_line node that includes the word 'terrible'
    #     (We look for the substring "terrible" in the text_line leaves.)
    "(text_line < /terrible/)",

    # 19) All emotion_line except NEUTRAL
    #     We do a negation: match emotion_line < /^(?!NEUTRAL).*/
    #     (Meaning, a child that doesn't start with 'NEUTRAL')
    "(emotion_line < /^(?!NEUTRAL).+/)",

    # 20) sibling check: text_line=txt has sibling rating_line=rat
    "(text_line=txt $ rating_line=rat)",
    
    # 21) Transitive child: "start << rating_line"
    #     Means: a 'start' node that has 'rating_line' as a descendant somewhere (not just direct child).
    "start << rating_line",

    # 22) Transitive parent: "rating_line >> start"
    #     Means: a 'rating_line' node whose ancestor is 'start' (should always match if rating_line is under start).
    "rating_line >> start",

    # 23) Negated child: "start !< rating_line"
    #     Means: a 'start' node that does NOT have a direct child 'rating_line'.
    #     (In your dataset, likely some have rating_line, some might not.)
    "start !< rating_line",

    # 24) Negated parent: "sentiment_line !> start"
    #     Means: a 'sentiment_line' node that does NOT have a direct parent labeled 'start'.
    #     That might rarely match if your grammar always places sentiment_line under start.
    "sentiment_line !> start",

    # 25) Transitive child capturing: "(start << rating_line=rat)"
    #     Finds a 'start' node that has a descendant 'rating_line' captured as 'rat'.
    #     Useful to see if you can capture the rating_line text.
    "(start << rating_line=rat)",

    # 26) "emotion_line !< sentiment_line"
    #     Means an 'emotion_line' that does NOT have direct child 'sentiment_line'.
    #     Typically an emotion_line won't have children in your grammar, so it might always match.
    "emotion_line !< sentiment_line",

    # 27) "sentiment_line >> start"
    #     Means a 'sentiment_line' that has 'start' as an ancestor.
    #     Usually that’s all sentiment_line nodes, but a good test for the '>>' operator.
    "sentiment_line >> start",

    # 28) Conjunction + transitive: "start & (start << emotion_line)"
    #     Means: a node labeled 'start' that also has 'emotion_line' as a descendant.
    #     Essentially the same as "start << emotion_line", but forced as an And-pattern.
    "(start & (start << emotion_line))",

    # 29) "emotion_line !< rating_line & (emotion_line >> start)"
    #     Means: an 'emotion_line' that does NOT have a direct child 'rating_line'
    #     AND has an ancestor 'start'. A more complex compound test.
    "(emotion_line !< rating_line & (emotion_line >> start))",

    # 30) Another capturing transitive example:
    #     "(text_line=txt << rating_line=rat)"
    #     Means 'text_line' that has 'rating_line' as a descendant in the tree 
    #     (not likely in your grammar, but good for testing).
    "(text_line=txt << rating_line=rat)",
]

# Finally, register under "sentiment"
register_dataset("sentiment", SentimentDataset)
