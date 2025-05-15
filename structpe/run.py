import argparse
import csv
import json
import os
import sys
from enum import Enum
import importlib
import logging
import warnings

logging.getLogger("structpe.utilities.llm_handler").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# We import the dataset registry and evaluation code:
from structpe.dataset.registry import get_dataset_class, DATASET_CLASSES, register_dataset
from structpe.evaluator.evaluator import Evaluator, compare_eval_results

# IMPORTANT: import your advanced pipeline function
from structpe.generator.generation import run_generation_pipeline

########################################################
#   TABLE DATASET DEFINITION (CSV/TSV)
########################################################

class TableSample:
    """
    A simple container class holding text, category, and label.
    The 'check_sample_constraints' function checks each sample to ensure certain minimal rules, like text length.
    """
    def __init__(self, text, category, label):
        self.text = text
        self.category = category
        self.label = label

    def check_sample_constraints(self, idx: int):
        # Example check: ensure that the 'text' field has >= 3 words
        if self.text:
            wc = len(self.text.split())
            if wc < 3:
                print(f"[TableSample] (idx={idx}) WARNING: text has {wc} words (<3).")


class TableDataset:
    """
    This dataset loads CSV/TSV files. The first row of the file is assumed to be a header like:
      text,category,label

    We store each sample as an instance of TableSample.
    """

    def __init__(self, file: str = None):
        self.samples = []
        print(f"[DEBUG] TableDataset __init__ called with file='{file}'")
        if file:
            self._load_from_csv_tsv(file)

    def _load_from_csv_tsv(self, filepath: str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"[TableDataset] CSV/TSV file not found: {filepath}")

        print(f"[DEBUG] TableDataset => _load_from_csv_tsv('{filepath}')")
        delimiter = "," if filepath.endswith(".csv") else "\t"
        with open(filepath, "r", encoding="utf-8") as f:
            lines = list(csv.reader(f, delimiter=delimiter))

        if not lines:
            print(f"[TableDataset] WARNING: file is empty => {filepath}")
            return

        header = lines[0]
        if len(header) < 3:
            raise ValueError(f"[TableDataset] CSV/TSV header must have >=3 cols. Found: {header}")

        for idx, row in enumerate(lines[1:], start=1):
            if len(row) < 3:
                print(f"[TableDataset] (row={idx}) WARNING: not enough cols => {row}")
                continue
            text, category, label = row[0], row[1], row[2]
            sample = TableSample(text, category, label)
            sample.check_sample_constraints(idx)
            self.samples.append(sample)

        print(f"[TableDataset] Loaded {len(self.samples)} samples from '{filepath}'.")


# Register the table dataset under the key "table"
register_dataset("table", TableDataset)

########################################################
#   load_dataset(...) FOR JSON / CSV / TSV
########################################################

def load_dataset(dataset_cls, dataset_name, file_path):
    print(f"[DEBUG] load_dataset called with dataset_cls='{dataset_cls.__name__}' dataset_name='{dataset_name}', file_path='{file_path}'")

    if not file_path:
        print("[DEBUG] No file_path provided, returning empty dataset_cls()")
        return dataset_cls()

    if not os.path.isfile(file_path):
        print(f"[Pipeline] WARNING: file not found => {file_path}")
        print("[DEBUG] Returning empty dataset_cls() because file is missing.")
        return dataset_cls()

    file_lower = file_path.lower()

    # For table dataset: .csv or .tsv
    if (file_lower.endswith(".csv") or file_lower.endswith(".tsv")) and dataset_name == "table":
        print(f"[DEBUG] Detected table CSV/TSV scenario => Using TableDataset.")
        ds = dataset_cls(file=file_path)
        print(f"[Pipeline] Loaded 'table' dataset from {file_path}, samples={len(ds.samples)}")
        return ds

    # For anything else, assume JSON
    if file_lower.endswith(".json"):
        print(f"[DEBUG] Detected .json => calling dataset_cls(file=file_path).")
        ds = dataset_cls(file=file_path)
        if hasattr(ds, "samples"):
            print(f"[Pipeline] Loaded dataset '{dataset_name}' from {file_path}, samples={len(ds.samples)}")
        else:
            print("[DEBUG] ds has no 'samples' attribute? Possibly 0 items.")
        return ds

    # Fallback
    print(f"[DEBUG] Fallback instantiation for dataset='{dataset_name}', file='{file_path}'")
    ds = dataset_cls(file=file_path)
    count = len(ds.samples) if hasattr(ds, "samples") else 0
    print(f"[Pipeline] (fallback) Loaded dataset '{dataset_name}' from {file_path}, samples={count}")
    return ds

########################################################
#   "do_generation" => calls run_generation_pipeline
########################################################

def do_generation(dataset_name, orig_samples, args):
    print("[DEBUG] do_generation => Using run_generation_pipeline for advanced LLM generation.")
    file_path = args.private_dataset_json
    file_type = "json"            # or "csv"/"tsv"
    concurrency = args.concurrency
    init_count = args.init_count
    iterations = args.iterations

    # NEW optional arguments:
    sim_mode   = getattr(args, "sim_mode", "avg")
    k          = getattr(args, "k", 2)
    l          = getattr(args, "l", 1)
    sigma      = getattr(args, "sigma", 0.1)
    temperature= getattr(args, "temperature", 0.7)
    top_p      = getattr(args, "top_p", 0.9)
    max_tokens = getattr(args, "max_tokens", 150)

    # Extra logging:
    print(f"[do_generation] => dataset_name='{dataset_name}', file='{file_path}', concurrency={concurrency}, init_count={init_count}, iterations={iterations}")
    print(f"[do_generation] => sim_mode={sim_mode}, k={k}, l={l}, sigma={sigma}, temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")

    synthetic_texts = run_generation_pipeline(
        file_path=file_path,
        file_type=file_type,
        concurrency=concurrency,
        init_count=init_count,
        iterations=iterations,
        endpoint="https://syndata.openai.azure.com/",
        deployment="gpt-4",
        dataset_name=dataset_name,
        sim_mode=sim_mode,
        k=k,
        l=l,
        sigma=sigma,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return synthetic_texts

########################################################
#   "do_evaluation"
########################################################

def do_evaluation(dataset_name, ds, label_for_print=None,
                  eval_json_out=None, original_dataset=None,
                  savedir=None, suffix=None):
    # 1) Determine the final_eval_json path
    if eval_json_out:
        base_name = os.path.basename(eval_json_out)
        out_dir = os.path.dirname(eval_json_out)
        if not out_dir:
            out_dir = savedir or ""
    else:
        base_name = "eval_results.json"
        out_dir = savedir or "."

    if suffix:
        name, ext = os.path.splitext(base_name)
        final_name = f"{name}_{suffix}{ext}"
    else:
        final_name = base_name

    final_eval_json = os.path.join(out_dir, final_name)

    print("\n[do_evaluation] BEGIN EVALUATION")
    print(f" - dataset_name: {dataset_name}")
    print(f" - label_for_print: {label_for_print}")
    print(f" - suffix: {suffix}")
    print(f" - final_eval_json => {final_eval_json}")
    if original_dataset:
        print(f" - original_dataset provided => used for embedding sim + KNN (samples={len(original_dataset.samples)})")
    else:
        print(" - NO original_dataset => KNN + embedding sim will have no reference data")
    print(f"[DEBUG] ds.samples length => {len(ds.samples) if hasattr(ds, 'samples') else 'no .samples?'}\n")

    # 2) Dynamically import the dataset module
    mod_path = f"structpe.dataset.{dataset_name}_dataset"
    dataset_module = None
    try:
        dataset_module = importlib.import_module(mod_path)
        print(f"[do_evaluation] Dynamically imported => {mod_path}")
    except Exception as e:
        print(f"[do_evaluation] WARNING: Could not import => {e}")

    # 3) Create the Evaluator, passing the module so the Verifier logs 'dataset_module=...'
    evaluator = Evaluator(
        dataset_name=dataset_name,
        original_dataset=original_dataset,
        dataset_module=dataset_module
    )

    # 4) Evaluate, also passing dataset_module again (optional but consistent):
    results_dict = evaluator.evaluate_full(
        dataset_obj=ds,
        dataset_module=dataset_module,
        eval_json_out=final_eval_json,
        savedir=savedir
    )

    if label_for_print:
        print(f"=== {label_for_print} => results in {final_eval_json} ===\n")
    print("[do_evaluation] DONE EVALUATION")

    return results_dict


########################################################
#   compare_data
########################################################

def compare_data(args):
    print(f"[DEBUG] compare_data => base='{args.savedir_base}', comp='{args.savedir_comp}', out='{args.savedir}'")
    base_dir = args.savedir_base
    comp_dir = args.savedir_comp
    out_dir = args.savedir
    os.makedirs(out_dir, exist_ok=True)

    base_jsons = [
        os.path.join(base_dir, f) for f in os.listdir(base_dir)
        if f.endswith(".json") and "eval_results" in f
    ]
    comp_jsons = [
        os.path.join(comp_dir, f) for f in os.listdir(comp_dir)
        if f.endswith(".json") and "eval_results" in f
    ]

    if not base_jsons:
        print(f"[Compare] No eval_results JSON in base directory => {base_dir}")
        return
    if not comp_jsons:
        print(f"[Compare] No eval_results JSON in comp directory => {comp_dir}")
        return

    base_file = base_jsons[0]
    comp_file = comp_jsons[0]
    print(f"[Compare] Using base: {base_file}")
    print(f"[Compare] Using comp: {comp_file}")

    with open(base_file, "r", encoding="utf-8") as bf:
        base_data = json.load(bf)
    with open(comp_file, "r", encoding="utf-8") as cf:
        comp_data = json.load(cf)

    compare_eval_results(base_data, comp_data, out_dir)
    print(f"[Compare] Done. Plots => {out_dir}")

########################################################
#   MAIN CLI
########################################################

def main_cli():
    print("[DEBUG] Entering main_cli() for structpe")
    parser = argparse.ArgumentParser(
        prog="structpe",
        description="Structpe CLI with generate, evaluate, compare, and list commands."
    )
    subparsers = parser.add_subparsers(dest="command", help="Top-level commands")

    list_parser = subparsers.add_parser("list", help="List functionality.")
    list_sub = list_parser.add_subparsers(dest="list_command")
    datasets_sub = list_sub.add_parser("datasets", help="List all registered datasets.")

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic data only.")
    generate_parser.add_argument("--private-dataset-name", required=True,
                                 help="Dataset name, e.g. 'sentiment','table','groundedqa'")
    generate_parser.add_argument("--private-dataset-json", default=None,
                                 help="Path to .json / .csv / .tsv file for the original dataset.")
    generate_parser.add_argument("--save-dir", required=True,
                                 help="Directory where we place 'synthetic.json'.")
    generate_parser.add_argument("--iterations", type=int, default=2,
                                 help="Number of iteration passes in generation.")
    generate_parser.add_argument("--init-count", type=int, default=100,
                                 help="Initial synthetic sample count.")
    generate_parser.add_argument("--concurrency", type=int, default=3,
                                 help="Max parallel requests for generation.")

    # NEW optional arguments for controlling advanced generation pipeline:
    generate_parser.add_argument("--sim-mode", default="avg", choices=["avg","max","min"],
                                 help="Similarity function for selection (avg|max|min).")
    generate_parser.add_argument("--k", type=int, default=2,
                                 help="Number of expansions (variation) per sample each iteration.")
    generate_parser.add_argument("--l", type=int, default=1,
                                 help="Number of times to replicate selected set before final variation.")
    generate_parser.add_argument("--sigma", type=float, default=0.1,
                                 help="Gaussian noise scale for sample fitness.")
    generate_parser.add_argument("--temperature", type=float, default=0.7,
                                 help="LLM temperature for generation calls.")
    generate_parser.add_argument("--top-p", type=float, default=0.9,
                                 help="LLM top_p for nucleus sampling.")
    generate_parser.add_argument("--max-tokens", type=int, default=150,
                                 help="LLM max tokens per generation call.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a dataset plus synthetic data.")
    evaluate_parser.add_argument("--private-dataset-name", required=True,
                                 help="Dataset name, e.g. 'sentiment','table','groundedqa'")
    evaluate_parser.add_argument("--private-dataset-json", default=None,
                                 help="Path to .json / .csv / .tsv for the *original* dataset.")
    evaluate_parser.add_argument("--synthetic-data", default=None,
                                 help="Path to .json / .csv / .tsv for synthetic data.")
    evaluate_parser.add_argument("--eval-json-out", default=None,
                                 help="Base name for results JSON; we append _synthetic, etc.")
    evaluate_parser.add_argument("--savedir", default=None,
                                 help="Directory for saving evaluation results & plots.")

    compare_parser = subparsers.add_parser("compare", help="Compare two sets of eval results.")
    compare_parser.add_argument("--private-dataset-name", required=True,
                                help="Dataset name, e.g. 'sentiment','table','groundedqa'")
    compare_parser.add_argument("--private-dataset-json", required=True,
                                help="(Not strictly used) path referencing the dataset.")
    compare_parser.add_argument("--savedir-base", required=True,
                                help="Dir with base eval_results*.json.")
    compare_parser.add_argument("--savedir-comp", required=True,
                                help="Dir with comparison eval_results*.json.")
    compare_parser.add_argument("--savedir", default="save_dir_compare",
                                help="Dir to store comparison plots.")

    args = parser.parse_args()

    print(f"[main_cli] Command => {args.command}")
    print(f"[main_cli] Full arguments => {args}")

    if args.command == "list":
        if args.list_command == "datasets":
            list_datasets()
        else:
            parser.print_help()

    elif args.command == "generate":
        generate_data(args)

    elif args.command == "evaluate":
        evaluate_piecemeal(args)

    elif args.command == "compare":
        compare_data(args)

    else:
        parser.print_help()


def list_datasets():
    print("[DEBUG] list_datasets => listing keys in DATASET_CLASSES")
    if not DATASET_CLASSES:
        print("No registered datasets.")
        return
    print("Registered datasets:")
    for dname in DATASET_CLASSES.keys():
        print(f" - {dname}")


def generate_data(args):
    print("[DEBUG] generate_data => Attempting to load original dataset.")
    dataset_name = args.private_dataset_name
    dataset_cls = get_dataset_class(dataset_name)
    print(f"[DEBUG] got dataset_cls='{dataset_cls.__name__}' from name='{dataset_name}'")

    # Load the original, if any
    orig_ds = None
    if args.private_dataset_json and os.path.isfile(args.private_dataset_json):
        orig_ds = load_dataset(dataset_cls, dataset_name, args.private_dataset_json)
        if hasattr(orig_ds, "samples"):
            print(f"[Generate] Original dataset => {len(orig_ds.samples)} samples.")
        else:
            print("[DEBUG] No .samples attribute in original dataset object??")
    else:
        print("[Generate] No original dataset loaded or file missing.")
        print(f"[DEBUG] private_dataset_json='{args.private_dataset_json}' isfile?={os.path.isfile(args.private_dataset_json) if args.private_dataset_json else 'NoneProvided'}")
        orig_ds = None

    # Actually call your advanced pipeline
    synthetic_texts = do_generation(dataset_name, orig_ds.samples if orig_ds else [], args)

    os.makedirs(args.save_dir, exist_ok=True)
    out_file = os.path.join(args.save_dir, "synthetic.json")
    with open(out_file, "w", encoding="utf-8") as f:
        for txt in synthetic_texts:
            item = {"text": txt, "generation": True}
            f.write(json.dumps(item) + "\n")
    print(f"[Generate] Wrote {len(synthetic_texts)} synthetic lines to '{out_file}'.")


def evaluate_piecemeal(args):
    print("[DEBUG] evaluate_piecemeal => checking original & synthetic files.")
    if not args.private_dataset_json or (not os.path.isfile(args.private_dataset_json)):
        raise ValueError(
            "[Evaluate] ERROR: You must supply a valid --private-dataset-json for the original dataset. "
            "File missing or not provided."
        )
    if not args.synthetic_data or (not os.path.isfile(args.synthetic_data)):
        raise ValueError(
            "[Evaluate] ERROR: You must supply a valid --synthetic-data for the synthetic dataset. "
            "File missing or not provided."
        )

    dataset_name = args.private_dataset_name
    dataset_cls = get_dataset_class(dataset_name)
    print(f"[DEBUG] dataset_name='{dataset_name}', dataset_cls='{dataset_cls.__name__}'")

    print("\n[evaluate_piecemeal] => Starting evaluation mode.\n")
    orig_ds = load_dataset(dataset_cls, dataset_name, args.private_dataset_json)
    if hasattr(orig_ds, "samples"):
        print(f"[Evaluate] Original dataset => {len(orig_ds.samples)} samples.")
    else:
        print("[DEBUG] Original dataset has no .samples. Possibly 0 items?")

    synth_ds = load_dataset(dataset_cls, dataset_name, args.synthetic_data)
    if hasattr(synth_ds, "samples"):
        print(f"[Evaluate] Synthetic dataset => {len(synth_ds.samples)} samples.")
    else:
        print("[DEBUG] Synthetic dataset has no .samples? Possibly 0 items?")

    print("\n[Evaluate] Evaluating the 'synthetic' dataset referencing the original...\n")
    do_evaluation(
        dataset_name=dataset_name,
        ds=synth_ds,
        label_for_print="Synthetic Dataset",
        eval_json_out=args.eval_json_out,
        savedir=args.savedir,
        suffix="synthetic",
        original_dataset=orig_ds
    )

    print("[Evaluate] Done with synthetic vs. original evaluation.\n")


if __name__ == "__main__":
    main_cli()
