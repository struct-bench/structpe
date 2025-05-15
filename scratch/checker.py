import json
import os
import argparse
from grammar import PARSERS

DEFAULT_INPUT_FILE = "test_data.json"
DEFAULT_REPORT_FILE = "report.json"

def load_test_data(file_path):
    """Loads test data from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file {file_path} not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_all_grammars(entry_text):
    """
    Validate a single text against ALL grammars.
    Returns a dict {grammar_name: "PASS"/"FAIL", ...}.
    """
    results = {}
    for grammar_name, parser in PARSERS.items():
        try:
            parser.parse(entry_text)
            results[grammar_name] = "PASS"
        except Exception:
            results[grammar_name] = "FAIL"
    return results

def validate_single_grammar(entry_text, grammar_name):
    """
    Validate a single text against ONE specific grammar.
    Returns "PASS" or "FAIL".
    """
    parser = PARSERS[grammar_name]
    try:
        parser.parse(entry_text)
        return "PASS"
    except Exception:
        return "FAIL"

def generate_report_all_grammars(test_data):
    """
    Generate a report for the scenario where we check each text against ALL grammars.
    Returns a dict with pass/fail totals across *all* grammars.
    """
    summary = {
        "total_entries": len(test_data),
        "details": [],
        "total_passes": 0,
        "total_fails": 0
    }

    for entry in test_data:
        text = entry.get("text", "")
        results = validate_all_grammars(text)

        # Count how many grammars passed/failed for this text
        passes = sum(1 for v in results.values() if v == "PASS")
        fails = sum(1 for v in results.values() if v == "FAIL")

        summary["total_passes"] += passes
        summary["total_fails"] += fails
        summary["details"].append({
            "text": text,
            "results": results
        })

    return summary

def generate_report_single_grammar(test_data, grammar_name):
    """
    Generate a report for checking each text against ONE grammar.
    Returns a dict with pass/fail totals for that grammar.
    """
    summary = {
        "grammar": grammar_name,
        "total_entries": len(test_data),
        "pass_count": 0,
        "fail_count": 0,
        "details": []
    }

    for entry in test_data:
        text = entry.get("text", "")
        status = validate_single_grammar(text, grammar_name)
        if status == "PASS":
            summary["pass_count"] += 1
        else:
            summary["fail_count"] += 1

        summary["details"].append({
            "text": text,
            "status": status
        })

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grammar Checker Script")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to the input JSON file (default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_FILE,
        help=f"Path to save the report JSON file (default: {DEFAULT_REPORT_FILE})"
    )
    parser.add_argument(
        "--grammar",
        type=str,
        help="Name of a specific grammar to check. If omitted, checks ALL grammars."
    )

    args = parser.parse_args()

    # Load data
    try:
        test_data = load_test_data(args.input)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # If user specified a grammar, verify it exists in PARSERS
    if args.grammar:
        if args.grammar not in PARSERS:
            print(f"Error: Grammar '{args.grammar}' not found in grammar.py")
            exit(1)
        print(f"Checking entries against grammar '{args.grammar}'...")
        report_data = generate_report_single_grammar(test_data, args.grammar)
    else:
        # No grammar specified -> check all grammars
        print("Checking entries against ALL grammars...")
        report_data = generate_report_all_grammars(test_data)

    # Save report
    try:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        print(f"Report generated: {args.report}")
    except Exception as e:
        print(f"Error writing report: {e}")
