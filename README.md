# structpe

### Burndown Chart

Assumptions: We characterize PE generations with Llama-8B and GPT4 only. We have a set of standardized metrics. We provide a set of 7 datasets chosen to evaluate the structural efficacy of DP synthetic data.   

| Task                   | Urgency        | Status      | ETA     |
|------------------------|----------------|-------------|---------|
| Onboard all 7 datasets (2S + 3A + 2V)   |  High      | In Progress | 4/15 EOD    |
| Vanilla PE numbers with LLama-8B on all 8 datasets (Generic Metrics) | High |  |   |
| Vanilla PE numbers with GPT-4 on all 8 datasets (Generic Metrics)      |  High    |      |     |
| Upload generation scripts for all 7 datasets with Llama8B + GPT4      |     High       |    |      |
| Report Vanilla PE Evaluations on 7 Datasets   |  High      | In Progress |   |
| Random API Baseline with the Llama-8B + GPT4 baselines     |     High       |    |      |
| Setup Leaderboard with the repo                       |    Low            |    Done         |         |
| Add a script that takes multiple json outputs & generates plots                       |    Low            |    Done         |         |
| Draft the Paper (https://www.overleaf.com/4391617511rqtsbrtknsdv#4a14ca)                      |    High            |    Done         |         |

**Hyperparameters for Generation**:
1. private dataset size: 3000
2. synthetic dataset size: 600 (shareGPT) / 300 (ICLR)
3. PE iterations: 10
4. variations per iteration: 2
5. blanking probability for Variation API: 0.7 (shareGPT) / 0.6 (ICLR)
6. privacy budget (epsilon): {1, 2, 4, \infty}

**PE for Real Dataset**:
1. Vanilla PE
2. PE with reformatting (self-debugging): reformatting variation samples before / after private voting process
3. PE with fixed format tokens in variation API: fix all / a subset of format tokens
4. PE with node extraction: extract different types of nodes
5. combinations of 2,3,4

**Illustration of Evaluation**:
1. For each generated dataset, show the evaluated results across iterations
2. For downstream task evaluations: show the averages prediction accuracy and standard error (std/\sqrt{n})


## Introduction
A flexible Python library for building, annotating, generating, and evaluating multiple dataset types with PE/variants.

This project implements a modular pipeline that supports:
1. Multiple **Dataset** classes (e.g. search queries, sentiment, conversation transcripts, Titanic data, ICLR reviews, hotel bookings).
2. Reflection-based **serialization** and **deserialization** with `_types.py` for enumerations and atomic classes.
3. A **generator** for producing synthetic data (currently demonstrated with `setiment` dataset).
4. An **evaluator** that can verify constraints and output detailed JSON metrics.

---
## Table of Contents
1. [Project Setup](#project-setup)
2. [Forking & Pull Requests](#forking--pull-requests)
3. [Adding a New Dataset](#adding-a-new-dataset)
4. [Running a Pipeline](#running-a-pipeline)
5. [Running Unit Tests](#running-unit-tests)
6. [Key Files & Directories](#key-files--directories)

---
## Project Setup
1. **Clone the Repo**
```bash
git clone https://github.com/vyraun/structpe.git
cd structpe
```

2. **Install**
- **Recommended**: Create a fresh virtual environment (`conda` or `venv`).
- Then install locally:
```bash
pip install .
```
- Or for editable mode (if you intend to develop and push changes):
```bash
pip install -e .
```

3. **Confirm Installation**
- After installing, run:
```bash
structpe list datasets
```
- You should see all registered datasets (e.g. `search_query`, `sentiment`, `iclr_review`, `conversation_dataset`, `titanic_dataset`, `hotel_booking`, etc.).

---
## Forking & Pull Requests
Contributions to **structpe** are welcome! To propose a change or new feature, follow these steps:
1. **Fork the Repo on GitHub**
   - Visit the [structpe GitHub page](https://github.com/yourusername/structpe), click “Fork”, and choose your GitHub account.

2. **Clone Your Fork**
```bash
git clone https://github.com/yourfork/structpe.git
cd structpe
```

3. **Create a Branch**
```bash
git checkout -b my-new-dataset
```

4. **Make Your Changes**
- Add new files, fix bugs, or implement new features.
- Update or add unit tests in `tests/`.

5. **Push & Open a Pull Request**
```bash
git commit -am "Add new dataset for XYZ"
git push origin my-new-dataset
```
- Then open a Pull Request on GitHub from your fork.

6. **Review & Merge**
- The maintainers will review your PR, offer feedback, and merge once approved.

---
## Adding a New Dataset
Structpe uses a **registry** pattern to easily integrate more datasets. Here’s how:
1. **Create a New File**
   - In `structpe/dataset/`, for example: `my_new_dataset.py`.
   - Define your sample class (`MyNewSample`) and a container class (`MyNewDataset`).
   - Use existing atomic types from `_types.py` or define constraints as needed.

2. **Register the Dataset**
   - At the end of that file:
```python
from structpe.dataset.registry import register_dataset
register_dataset("my_new_dataset", MyNewDataset)
```
   - Now you can do `structpe run --dataset-name=my_new_dataset`.

3. **Update run.py** (Optional)
   - If you want a default sample for demonstration:
```python
elif dataset_name == "my_new_dataset":
    from structpe.dataset.my_new_dataset import MyNewSample
    ds.add_sample(MyNewSample(...))
```
   - In `pipeline_command`, parse the JSON fields for your new sample.

4. **Test**
   - Add or modify a test in `tests/test_dataset.py` to ensure everything works.

---
## Running a Pipeline
Structpe’s CLI has a `pipeline` command that reads from a JSON file describing the dataset and samples.
1. **Create a JSON** (e.g. `my_pipeline.json`):
```json
{
  "dataset_name": "search_query",
  "samples": [
    {
      "query_text": "buy shoes",
      "intent": "TRANSACTIONAL",
      "topic": "SHOPPING",
      "word_count": 2
    },
    {
      "query_text": "flight deals",
      "intent": "NAVIGATIONAL",
      "topic": "TRAVEL",
      "word_count": 2
    }
  ]
}
```

2. **Run the Pipeline**:
```bash
structpe pipeline --json-file=my_pipeline.json
```

3. **(Optional) Write Evaluation Stats to JSON**
```bash
structpe pipeline --json-file=my_pipeline.json --eval-json-out=my_metrics.json
```
- The evaluator will output a JSON file with distribution stats, constraint failures, etc.

---
## Running Unit Tests
Structpe includes tests under `tests/`.
1. **Install** in your environment (or editable mode):
```bash
pip install -e .
```
2. **Run All Tests**:
```bash
python -m unittest discover tests
```
3. **Run a Specific Test**:
```bash
python -m unittest tests.test_dataset
```
4. **Add New Tests**:
   - In `tests/`, create or update `test_*.py` files to cover your changes or new datasets.

---
## Key Files & Directories
- **`structpe/_types.py`**  
  Holds enumerations and atomic range classes (e.g. `AtomicRangeInt`) used by multiple datasets.

- **`structpe/dataset/`**  
  Holds each dataset definition (`search_dataset.py`, `hotel_booking_dataset.py`, etc.) plus `registry.py` for dynamic dataset lookup.

- **`structpe/descriptor/descriptor.py`**  
  Implements reflection-based serialization so that entire dataset objects can be stored as JSON and reconstructed.

- **`structpe/evaluator/`**  
  Contains the `Evaluator` class (with JSON output) and supporting classes (`LLMJudge`, `Verifier`, etc.) for constraint checks, distribution stats, and more.

- **`structpe/generator/generation.py`**  
  Demonstrates how to create synthetic samples from existing dataset descriptions (currently for `search_query`).

- **`structpe/run.py`**  
  Houses the CLI. Subcommands:
    - `list datasets`: Show registered datasets
    - `run --dataset-name=XYZ`: Instantiate and evaluate a dataset
    - `pipeline --json-file=FILE`: Build a dataset from JSON samples, then evaluate

- **`tests/`**  
  Contains unit tests such as:
    - `test_dataset.py` (checks correctness of dataset classes),
    - `test_pipeline.py` (verifies pipeline logic),
    - `test_evaluator.py` (tests evaluation output).

---
