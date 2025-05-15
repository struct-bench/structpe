# Evaluation Framework Documentation

This document provides an overview of the existing metrics in the evaluation framework and explains how to add a new metric (like BLEU or ChrF). This is intended to guide you through the structure of the evaluator code and the steps needed to integrate your own scoring logic.

## 1. Overview of Existing Metrics

### 1.1 LLM Correctness (1..5 Scale)
- **Location**: Implemented via `LLMJudge` in `evaluator_types.py`.
- **Purpose**: Calls a large language model (LLM) to produce a numeric score from 1..5 reflecting how "correct" or "high quality" a sample is.
- **Storage**: In the final JSON results, stored under `"llm_judge" -> "correctness_score"` for each sample.

### 1.2 Embedding Similarity ([0..1] Range)
- **Location**: Implemented in the `Metrics` class, using `sentence_transformers` to embed text, then compute max cosine similarity with original dataset texts.
- **Purpose**: Measures how similar a newly generated text is to any original text. Yields a number between 0.0 (completely dissimilar) and 1.0 (identical).
- **Storage**: In the final JSON results, stored under `"metrics" -> "embedding_similarity"` for each sample.

### 1.3 Graph Adjacency (1..5 Scale)
- **Location**: Implemented in `GraphEvaluator` and `GraphLLMJudge`.
- **Purpose**: Uses an LLM to check the sample’s adjacency constraints (e.g., referencing the reversed adjacency definition). Assigns a numeric score from 1..5.
- **Storage**: In the final JSON results, under `"graph_evaluator" -> "graph_score"` and a `"graph_message"` for more detail.

### 1.4 Grammar Check (Pass/Fail + Debug)
- **Location**: `GenericGrammarCheck` dynamically loads a grammar from the dataset module (if provided).
- **Purpose**: Parses a sample’s multiline representation. If it fails to parse, the sample fails. If it parses, the sample passes the grammar check.
- **Storage**: Boolean pass/fail is stored under `"grammar_check" -> "passed"`. There is also debug info and an optional similarity matrix of each line in the grammar representation.

### 1.5 Dataset-Level Verification
- **Location**: `DatasetLevelVerifier` in `evaluator_types.py`.
- **Purpose**: Checks constraints spanning all samples, e.g., distribution constraints, average rating checks, etc. The dataset module can define `verify_dataset(...)` returning a list of `(index, pass_bool, reason)`.
- **Storage**: Results are stored under `"aggregate" -> "dataset_verifier" -> { used, passed_count, failed_count, failures }`.

---

## 2. Structure of the Evaluation Repository

### `evaluator_types.py`
Contains classes for each metric/check:
- `LLMJudge`, `Verifier`, `Metrics`, `GraphEvaluator`, `GenericGrammarCheck`, and `DatasetLevelVerifier`.

### `evaluator.py`
Orchestrates the evaluation by:
- Instantiating the various metric classes.
- Iterating over each sample.
- Collecting metric results in a results dictionary.
- Saving JSON and producing visual plots.

#### `compare_eval_results(...)`
Compares two sets of evaluation results to produce side-by-side charts.

### Dataset-Specific Files
For example, `sentiment_dataset.py`:
- Defines how data is loaded into sample objects, adjacency constraints, optional grammar, etc.

---

## 3. Adding a New Metric (e.g., BLEU or ChrF)

This step-by-step guide shows how to introduce a new metric in the existing framework so it is computed and reported alongside the existing ones.

### Step 1: Implement a Class for the New Metric
Create a new class (e.g., `BleuMetric`) within `evaluator_types.py` (or a separate file you can import). This class should:
- Initialize any model/resources needed to compute your metric (e.g., BLEU scoring library, ChrF library).
- Provide a method (e.g., `score_sample`) that takes a synthetic sample (or references) and returns a numeric score.

#### Example:
```python
class BleuMetric:
    def __init__(self, original_dataset):
        # Possibly store references, or load a BLEU library
        self.original_dataset = original_dataset
        self.reference_texts = []
        # ... gather references from original_dataset ...

    def score_sample(self, synthetic_sample):
        """
        Compute the BLEU score for the sample against reference_texts.
        Return a float. If something fails, return 0.0.
        """
        try:
            candidate = getattr(synthetic_sample, "text", "")
            # ... compute BLEU with your library ...
            return bleu_score
        except Exception as e:
            print(f"[BleuMetric] WARNING: {e}")
            return 0.0

    def to_debug_dict(self, synthetic_sample):
        """
        Return an optional dict with debug information
        (e.g. intermediate calculation steps, reference matches).
        """
        bleu_val = self.score_sample(synthetic_sample)
        return {
            "bleu_value": bleu_val,
            "details": "Any intermediate steps or reference lines"
        }
```

### Step 2: Instantiate Your Metric in the Evaluator
In `Evaluator.__init__`, add logic to create your new metric class, similarly to how `Metrics` is created. For example:
```python
self.bleu_metric = None
try:
    from structpe.evaluator.evaluator_types import BleuMetric
    self.bleu_metric = BleuMetric(original_dataset)
except Exception as e:
    print(f"[Evaluator] WARNING: BLEU Metric init => {e}")
    self.bleu_metric = None
```

### Step 3: Integrate the Metric Calculation in `_evaluate_dataset`
Within `Evaluator._evaluate_dataset(...)`, you’ll see loops over each sample. Collect metric results for each sample and store them in `samples_results[str(idx)]`. For example:
```python
bleu_val = 0.0
bleu_debug = {}
if self.bleu_metric:
    try:
        bleu_val = self.bleu_metric.score_sample(sample)
        bleu_debug = self.bleu_metric.to_debug_dict(sample)
    except Exception as e:
        print(f"[Evaluator] WARNING: BLEU => {e}")
        bleu_val = 0.0
        bleu_debug = {"bleu_value": 0.0, "error": str(e)}

# Store the score in the final result
samples_results[str(idx)]["bleu_score"] = {
    "bleu_value": bleu_val,
    "trace_debug": bleu_debug
}
```

### Step 4: Aggregate the New Metric
At the end of `_evaluate_dataset(...)`, aggregate metrics in `results_dict["aggregate"]`. For example:
```python
# Initialize aggregator sums
total_bleu = 0.0

# Inside the main loop, accumulate bleu_val
total_bleu += bleu_val

# After the loop, compute average
avg_bleu = round(total_bleu / total, 3)

# Insert it in results_dict
results_dict["aggregate"]["bleu_metric"] = {
    "average_bleu": avg_bleu
}
```

### Step 5: Plot or Compare Your Metric (Optional)
To generate a histogram or bar chart:
```python
# 1) Collect a list of BLEU values
bleu_values = [ samples_results[str(i)]["bleu_score"]["bleu_value"] for i in range(total) ]

# 2) Plot in `_generate_plots(...)`:
if bleu_values:
    plt.figure(figsize=(6,4))
    plt.hist(bleu_values, bins=10, edgecolor='black')
    plt.title("BLEU Score Distribution")
    plt.xlabel("BLEU Score")
    plt.ylabel("Count")
    plt.tight_layout()
    png_path = os.path.join(results_dir, "hist_bleu_scores.png")
    plt.savefig(png_path)
    plt.close()
    print(f"[Evaluator] Saved BLEU score histogram to {png_path}")
```

### Step 6: Validate and Extend
- **Validation**: Run the evaluator with a known dataset to verify that the new metric is properly calculated.
- **Extensibility**: Introduce other metrics (ChrF, ROUGE, etc.) by following this same pattern:
  - Create a class that loads references and computes the metric.
  - Add it to the evaluator (`__init__` + `_evaluate_dataset`) to store per-sample results and aggregator stats.
  - Optionally produce specialized plots.

---

## 4. Summary
- **Existing metrics** include LLM correctness, embedding similarity, graph adjacency, grammar checks, and dataset-level constraints.
- **Adding a new metric** (BLEU, ChrF, etc.) involves:
  1. Creating a metric class with a scoring function.
  2. Instantiating it in the evaluator.
  3. Computing the metric for each sample in `_evaluate_dataset(...)`.
  4. Storing results in the JSON output and aggregator.
  5. (Optional) Generating plots.

Following these steps ensures that your custom metric is seamlessly integrated with all other evaluator features, including side-by-side comparisons and final reporting.