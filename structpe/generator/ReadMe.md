# README: Synthetic Text Generation with PE Library and Azure OpenAI

## 1. Overview

This module (`generation.py`) provides:

### Data Handling (`JsonData`)
- Subclass of `pe.data.Data` that allows the PE pipeline to work with your data.
- Provides standard methods (`filter_label_id()`, `merge()`, `save_checkpoint()`, etc.).
- **Loading JSON, CSV, or TSV**:
    - `load_json_as_dataset()`, `load_csv_as_dataset()`, `load_tsv_as_dataset()`, and a unified `load_data_as_dataset()` function.
    - Assigns a single label (`PE.LABEL_ID = 0`) in all rows for simplicity.
    - Populates `metadata["label_info"]` with a minimal `LabelInfoObject()`.

### Azure OpenAI LLM (`MyAzureOpenAILLM`)
- Uses `DefaultAzureCredential` + Azure OpenAI to avoid manual `AZURE_OPENAI_API_KEY` usage.
- Handles concurrency via `ThreadPoolExecutor` and retry logic with `Tenacity`.

### Main Generation Pipeline (`run_generation_pipeline`)
- Accepts file path + file type (`json`, `csv`, or `tsv`).
- Loads data, builds an LLM pipeline (`LLMAugPE`, `PEPopulation`, etc.), runs multiple iterations, and saves final synthetic text to a CSV.

### `SampleGenerator` Class
- Can be directly imported in scripts like `run.py`.
- Expects a list of private records (`private_data`) and orchestrates the same pipeline, returning the final synthetic texts as a Python list.

---

## 2. File Input Options

### JSON
- Use `load_json_as_dataset(file_path)` or call `load_data_as_dataset(file_path, file_type="json")`.
- The module expects a top-level list of JSON objects or strings.

### CSV
- Use `load_csv_as_dataset(file_path)` or call `load_data_as_dataset(file_path, file_type="csv")`.
- Expects columns. If no `text` column is found, all columns are merged into a single `text` field.

### TSV
- Similarly, `load_tsv_as_dataset(file_path)` or `load_data_as_dataset(file_path, file_type="tsv")`.
- Uses `delimiter = "\t"`.
- Each record in the final DataFrame will have `text` and `PE.LABEL_ID` columns.

---

## 3. How to Use `run_generation_pipeline`

```python
from structpe.generator.generation import run_generation_pipeline

synthetic_texts = run_generation_pipeline(
        file_path="data/input.tsv",
        file_type="tsv",   # or "csv" or "json"
        concurrency=4,
        init_count=10,
        iterations=3,
        endpoint="https://myazureendpoint.openai.azure.com/",
        deployment="gpt-4"
)
```

### Parameters:
- **`file_path`**: Path to your input file (`JSON`, `CSV`, or `TSV`).
- **`file_type`**: Must be `"json"`, `"csv"`, or `"tsv"`.
- **`concurrency`**: Number of threads to use for Azure OpenAI calls.
- **`init_count`**: Initial sample count.
- **`iterations`**: How many iteration cycles.
- **`endpoint`**: Your Azure OpenAI endpoint.
- **`deployment`**: Name of the model deployment (e.g., `"gpt-4"`).

### Return Value:
A list of final generated strings.

---

## 4. Using `SampleGenerator` in Another Script

```python
from structpe.generator.generation import SampleGenerator

# Suppose 'private_data' is a list of dicts, each with "text".
gen = SampleGenerator(
        concurrency=4,
        init_count=5,
        iterations=2,
        endpoint="https://myazureendpoint.openai.azure.com/",
        deployment="gpt-4"
)

results = gen.generate(private_data=[
        {"text": "Hello world!"},
        {"text": "Some input line"},
        {"any_key": "No text field? We'll just convert it to text anyway."}
])
print("Generated synthetic texts:", results)
```

### Notes:
- The constructor args match those of `run_generation_pipeline`.
- **`.generate(...)` method**:
    - Builds a `JsonData` from `private_data`.
    - Sets up the pipeline with `MyAzureOpenAILLM`.
    - Runs multiple iterations, saves a final CSV, and returns a list of synthetic text strings.

---

## 5. Implementation Details

### Label Info
- Each dataset or snippet uses a single label `ID = 0`.
- If you have multiple labels, you can adapt `LABEL_ID_COLUMN_NAME` accordingly.

### Azure Authentication
- `MyAzureOpenAILLM` uses `DefaultAzureCredential` from `azure.identity`.
- Ensure you’re logged in or have your Azure environment credentials set so that your script can fetch tokens automatically.

### Callbacks and Logging
- The pipeline uses `SaveCheckpoints`, `SaveTextToCSV`, `CSVPrint`, and `LogPrint` from the PE library.
- All logs, synthetic CSVs, and checkpoint data are stored in a `results/` directory.

### Parallel Requests & Retries
- The LLM calls are made in parallel with a thread pool.
- We also employ `tenacity` to retry on transient network or rate-limit errors.

---

## 6. Summary

- `JsonData` is the base data container for the PE pipeline.
- `load_*_as_dataset` methods parse your input into a consistent format, all labeled with `PE.LABEL_ID=0`.
- `MyAzureOpenAILLM` authenticates via Azure tokens and calls the deployed LLM.
- `run_generation_pipeline` orchestrates all steps from loading data to returning final synthetic strings.
- `SampleGenerator` is a convenient class to do the same if you have an in-memory list of private records.

Now you can generate synthetic text from JSON, CSV, or TSV using Azure OpenAI in a single pass, with minimal environment setup.