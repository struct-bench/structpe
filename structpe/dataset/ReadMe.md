# Dataset Evaluation Framework

This framework provides a unified pipeline for evaluating custom datasets that contain both text and structured labels. It performs:

- Loading dataset files into custom classes.
- Verifying each sample (word counts, rating bounds, adjacency constraints, grammar parsing).
- Scoring each sample with LLM-based correctness, adjacency checks, and embedding similarity.
- Computing optional KNN-based metrics (precision/recall).
- Aggregating results into JSON and generating plots.
- Running `dataset_metric_*` functions, which can be either:
    - **dataset-level** (run once) or
    - **sample-level** (run per sample), storing the final outputs in the aggregator or per-sample sections.

Below is a concise guide, using the `sentiment_dataset` as an example.

---

## 1. Installing & Basic Usage

Install the library in editable mode:

```bash
pip install -e .
```

Run the CLI to evaluate your dataset:

```bash
structpe evaluate \
  --private-dataset-name=sentiment \
  --private-dataset-json=data/sentiment.json \
  --synthetic-data=data/sentiment.json \
  --savedir base_run
```

This reads `sentiment.json`, runs the entire pipeline, and writes results + plots to `base_run/`.

---

## 2. Key Concepts & Flow

### Dataset Loader

Each dataset (e.g. `SentimentDataset`) is registered with:

```python
from structpe.dataset.registry import register_dataset
register_dataset("sentiment", SentimentDataset)
```

When you run `structpe evaluate --private-dataset-name=sentiment`, the framework dynamically loads `SentimentDataset`, which:

- Parses your JSON into sample objects.
- Checks constraints per sample (like rating bounds).

### LLM Judges

The Evaluator calls:

- `LLMJudge` for correctness (score in `[1..5]`).
- `GraphEvaluator` for adjacency checks (also `[1..5]`).
- Metrics to compute text embeddings & measure similarity.

### Grammar Checks

If your dataset provides:

```python
def get_grammar():
        return """some grammar string"""
```

And:

```python
def build_grammar_string_for_check(sample):
        # convert your sample to multiline text that the grammar can parse
        ...
```

Then `GenericGrammarCheck` tries parsing each sample. If it fails, that sample is marked as grammar-failed.

### Dataset-Level Verification

Optionally, your dataset can define:

```python
def verify_dataset(self):
        # e.g. check rating distributions or label frequencies
        return [(index, pass_bool, reason), ...]
```

These results get appended to the final aggregator under `dataset_verifier`.

---

## 3. Dataset Metrics (NEW)

You can define arbitrary metrics for your dataset inside your `*_dataset.py` file. Mark them with a special decorator:

```python
from structpe.dataset.decorators import dataset_metric  # or wherever you define it

@dataset_metric(level="dataset")
def dataset_metric_rating_distribution(dataset_obj, all_samples_results):
        # Runs once for the entire dataset
        # Return a dict with any info you like
        ...
        return {
            "rating_distribution": {...},
            "num_rated_samples": ...
        }

@dataset_metric(level="sample")
def dataset_metric_score_each_sample(dataset_obj, single_sample_result):
        # Runs once per sample
        # single_sample_result["raw_sample_obj"] is your actual sample object
        sample_obj = single_sample_result["raw_sample_obj"]
        ...
        return {"some_metric": ...}
```

### How it Works:

- **`level="dataset"`**: The framework calls your function once at the end, passing `(dataset_obj, all_samples_results)`. You can compute any global stats and return a dict that goes in `aggregate.dataset_dependent_metrics["dataset_metrics"][fn_name]`.
- **`level="sample"`**: The framework calls your function for each sample. The result is stored in `samples_results[str(idx)]["sample_dependent_metrics"][fn_name]`.

In the final JSON, you’ll see:

```json
"dataset_dependent_metrics": {
    "dataset_metrics": {
        "dataset_metric_rating_distribution": {
            "rating_distribution": {...},
            "num_rated_samples": ...
        }
    },
    "sample_metrics": "Populated inside each sample's sample_dependent_metrics"
}
```

And each sample has something like:

```json
{
    "sample_dependent_metrics": {
        "dataset_metric_score_each_sample": {
            "some_metric": ...
        }
    }
}
```

---

## 4. `compute_node_similarities`

Additionally, if your dataset defines a list of node pairs in `compute_node_similarities`, e.g.:

```python
compute_node_similarities = [
        ("sentiment", "emotion"),
        ("rating", "text")
]
```

Then whenever grammar parsing is successful, the evaluator:

1. Parses each sample into lines (with `build_grammar_string_for_check`).
2. Embeds them.
3. Constructs a NxN cosine similarity matrix among the lines.

For each `(left_name, right_name)` pair in `compute_node_similarities`, the evaluator:

- Looks up the line index for `left_name` and `right_name`.
- Extracts the similarity from the NxN matrix.
- Stores it in e.g. `"left_name-right_name-similarity": 0.8123` in `sample["grammar_check"]`.

This can help you see how “text” relates to “rating,” or how “emotion” lines relate to “sentiment” lines, etc.

---

## 5. Steps to Onboard Your Own Dataset

1. Create `my_new_dataset.py` with a class `MyDataset` that loads JSON and does checks.
2. Define any grammar or adjacency if relevant.
3. Register your dataset:

     ```python
     from structpe.dataset.registry import register_dataset
     register_dataset("my_dataset", MyDataset)
     ```

4. (Optional) Provide `verify_dataset(...)` for global checks.
5. (Optional) Provide any `dataset_metric(level=...)` functions to compute custom metrics, either dataset- or sample-level.
6. (Optional) If lines in grammar have fields that are logically comparable, define `compute_node_similarities = [("fieldA", "fieldB"), ...]`.

Then run:

```bash
structpe evaluate \
        --private-dataset-name=my_dataset \
        --private-dataset-json=data/my_dataset.json \
        --savedir results_my_dataset
```

You’ll get a comprehensive JSON summarizing correctness, adjacency, grammar, KNN-based metrics, plus your custom dataset metrics.

---

That’s it! You now have:

- Dataset-level & sample-level metrics via `dataset_metric(level=...)`.
- Optional node similarity checks with `compute_node_similarities`.
- A standard pipeline for verifying, scoring, and analyzing your dataset.
