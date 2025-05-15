"""
run.py: Provides the CLI commands for structpe:
 - list datasets
 - run --dataset-name=...
 - pipeline --json-file=...
 
Allows optional argument --eval-json-out to store evaluation metrics in a JSON file.
"""

import argparse
import json
import sys
import os

from structpe.dataset.registry import get_dataset_class, DATASET_CLASSES
from structpe.descriptor.descriptor import DatasetDescriptor
from structpe.generator.generation import SampleGenerator
from structpe.evaluator.evaluator import Evaluator

def main_cli():
    """
    Entry point for the 'structpe' command. This function sets up argparse
    subcommands for 'list', 'run', and 'pipeline'.
    """
    parser = argparse.ArgumentParser(prog="structpe", description="Structpe CLI (JSON pipeline)")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Subcommand: list
    list_parser = subparsers.add_parser("list", help="Listing functionality.")
    list_subparsers = list_parser.add_subparsers(dest="list_command")
    list_datasets_parser = list_subparsers.add_parser("datasets", help="List all registered datasets.")

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run a quick dataset pipeline.")
    run_parser.add_argument("--dataset-name", type=str, required=True, help="Name of registered dataset.")
    run_parser.add_argument("--eval-json-out", type=str, default=None,
                            help="If provided, path to a JSON file where evaluation results will be written.")

    # Subcommand: pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Run a pipeline from a JSON file.")
    pipeline_parser.add_argument("--json-file", type=str, required=True, help="Path to pipeline JSON definition.")
    pipeline_parser.add_argument("--eval-json-out", type=str, default=None,
                                 help="If provided, path to a JSON file for storing evaluation results.")

    # Parse arguments and dispatch to sub-commands
    args = parser.parse_args()

    if args.command == "list":
        if args.list_command == "datasets":
            list_datasets()
        else:
            parser.print_help()

    elif args.command == "run":
        run_command(args.dataset_name, args.eval_json_out)

    elif args.command == "pipeline":
        pipeline_command(args.json_file, args.eval_json_out)

    else:
        parser.print_help()

def list_datasets():
    """
    Lists all dataset names that have been registered in dataset/registry.py
    """
    if not DATASET_CLASSES:
        print("No datasets registered.")
        return

    print("Registered datasets:")
    for name in DATASET_CLASSES:
        print(" -", name)

def run_command(dataset_name: str, eval_json_out: str = None):
    """
    Quick pipeline for the given dataset name:
     1) Instantiate dataset from the registry
     2) Add a default sample if recognized
     3) Serialize the dataset
     4) Optionally generate a new sample (only for search_query)
     5) Evaluate the dataset, and if eval_json_out is given, write stats to JSON
    """
    # 1) Instantiate dataset
    dataset_cls = get_dataset_class(dataset_name)
    ds = dataset_cls()

    # 2) Add default sample if recognized
    if dataset_name == "search_query":
        from structpe.dataset.search_dataset import SearchSample
        from structpe._types import QueryIntent, QueryTopic, AtomicRangeInt
        ds.add_sample(SearchSample("book flights",
                                   QueryIntent.NAVIGATIONAL,
                                   QueryTopic.TRAVEL,
                                   AtomicRangeInt(2,1,10)))
    elif dataset_name == "sentiment":
        from structpe.dataset.sentiment_dataset import SentimentSample
        from structpe._types import SentimentLabel
        ds.add_sample(SentimentSample("I love this product", SentimentLabel.POSITIVE))
    elif dataset_name == "iclr_review":
        from structpe.dataset.iclr_review_dataset import (
            ICLRPaperReviewSample, ICLRComment, ICLRCommentRole, ICLRRecommendation
        )
        from structpe._types import AtomicRangeInt, AtomicRangeFloat
        ds.add_sample(ICLRPaperReviewSample(
            paper_title="Transformers Revisited",
            rating=AtomicRangeInt(7, 1, 10),
            confidence=AtomicRangeFloat(4.5, 1.0, 5.0),
            recommendation=ICLRRecommendation.WEAK_ACCEPT,
            comments=[
                ICLRComment("Promising approach but missing ablations.", ICLRCommentRole.REVIEWER),
                ICLRComment("Will add experiments in revision.", ICLRCommentRole.AUTHOR),
            ]
        ))
    elif dataset_name == "conversation_dataset":
        from structpe.dataset.conversation_dataset import (
            ConversationSample, ConversationTurn, ConversationRole
        )
        ds.add_sample(ConversationSample([
            ConversationTurn(ConversationRole.USER, "Hello! I'd like a pizza."),
            ConversationTurn(ConversationRole.ASSISTANT, "Sure, which toppings do you want?"),
        ]))
    elif dataset_name == "titanic_dataset":
        from structpe.dataset.titanic_dataset import TitanicSample, SexEnum
        from structpe._types import AtomicRangeInt, AtomicRangeFloat
        ds.add_sample(TitanicSample(
            name="John Smith",
            age=AtomicRangeInt(30, 0, 120),
            sex=SexEnum.MALE,
            fare=AtomicRangeFloat(10.5, 0.0, 600.0),
            survived=True
        ))
    elif dataset_name == "hotel_booking":
        from structpe.dataset.hotel_booking_dataset import HotelBookingSample
        from structpe._types import AtomicRangeInt, AtomicRangeFloat
        # Add a valid default sample
        ds.add_sample(HotelBookingSample(
            booking_id="BKG-001",
            check_in_date="2025-03-10",
            check_out_date="2025-03-12",
            num_guests=AtomicRangeInt(2, 1, 20),
            total_price=AtomicRangeFloat(350.0, 0.0, 5000.0)
        ))
    else:
        print(f"Unrecognized dataset '{dataset_name}'; no default samples added.")

    # 3) Serialize dataset
    descriptor = DatasetDescriptor()
    serialized_ds = descriptor.serialize(ds)
    print("=== Serialized Dataset ===")
    print(serialized_ds)

    # 4) Generate new sample only if "search_query"
    if dataset_name == "search_query":
        generator = SampleGenerator()
        new_sample_json = generator.generate_new_sample(serialized_ds)
        print("\n=== Generated New Sample ===")
        print(new_sample_json)
    else:
        print("\n(No generator logic for this dataset in 'run' command.)")

    # 5) Evaluate dataset, possibly writing a JSON file
    evaluator = Evaluator()
    metrics = evaluator.evaluate_and_write_json(ds, output_json=eval_json_out)
    print("\n=== Evaluation Metrics (Dict) ===")
    print(metrics)

    if eval_json_out:
        print(f"Saved evaluation results to {eval_json_out}")

def pipeline_command(json_file: str, eval_json_out: str = None):
    """
    Pipeline command: reads a JSON file specifying dataset_name and samples,
    builds & evaluates the dataset. If eval_json_out is provided, writes results to a JSON file.
    """
    if not os.path.isfile(json_file):
        print(f"JSON file '{json_file}' not found.")
        sys.exit(1)

    with open(json_file, "r") as f:
        pipeline_config = json.load(f)

    dataset_name = pipeline_config.get("dataset_name")
    if not dataset_name:
        print("No 'dataset_name' in the JSON pipeline config.")
        sys.exit(1)

    dataset_cls = get_dataset_class(dataset_name)
    ds = dataset_cls()

    samples_list = pipeline_config.get("samples", [])

    # Handle each known dataset type
    if dataset_name == "search_query":
        from structpe.dataset.search_dataset import SearchSample
        from structpe._types import QueryIntent, QueryTopic, AtomicRangeInt
        for sconf in samples_list:
            ds.add_sample(SearchSample(
                query_text=sconf["query_text"],
                intent=QueryIntent(sconf["intent"]),
                topic=QueryTopic(sconf["topic"]),
                word_count=AtomicRangeInt(sconf["word_count"], 1, 100)
            ))
    elif dataset_name == "sentiment":
        from structpe.dataset.sentiment_dataset import SentimentSample
        from structpe._types import SentimentLabel
        for sconf in samples_list:
            ds.add_sample(SentimentSample(
                text=sconf["text"],
                sentiment=SentimentLabel(sconf["sentiment"])
            ))
    elif dataset_name == "iclr_review":
        from structpe.dataset.iclr_review_dataset import (
            ICLRPaperReviewSample, ICLRComment, ICLRCommentRole, ICLRRecommendation
        )
        from structpe._types import AtomicRangeInt, AtomicRangeFloat
        for sconf in samples_list:
            comment_objs = []
            for cconf in sconf.get("comments", []):
                role = ICLRCommentRole(cconf["role"])
                comment_objs.append(ICLRComment(cconf["text"], role))

            ds.add_sample(ICLRPaperReviewSample(
                paper_title=sconf["paper_title"],
                rating=AtomicRangeInt(sconf["rating"], 1, 10),
                confidence=AtomicRangeFloat(sconf["confidence"], 1.0, 5.0),
                recommendation=ICLRRecommendation(sconf["recommendation"]),
                comments=comment_objs
            ))
    elif dataset_name == "conversation_dataset":
        from structpe.dataset.conversation_dataset import (
            ConversationSample, ConversationTurn, ConversationRole
        )
        for sconf in samples_list:
            turn_objs = []
            for tc in sconf["turns"]:
                role = ConversationRole(tc["role"])
                turn_objs.append(ConversationTurn(role, tc["text"]))
            ds.add_sample(ConversationSample(turn_objs))
    elif dataset_name == "titanic_dataset":
        from structpe.dataset.titanic_dataset import TitanicSample, SexEnum
        from structpe._types import AtomicRangeInt, AtomicRangeFloat
        for sconf in samples_list:
            sex_val = SexEnum(sconf["sex"])
            ds.add_sample(TitanicSample(
                name=sconf["name"],
                age=AtomicRangeInt(sconf["age"], 0, 120),
                sex=sex_val,
                fare=AtomicRangeFloat(sconf["fare"], 0.0, 600.0),
                survived=sconf["survived"]
            ))
    elif dataset_name == "hotel_booking":
        from structpe.dataset.hotel_booking_dataset import HotelBookingSample
        from structpe._types import AtomicRangeInt, AtomicRangeFloat
        for sconf in samples_list:
            ds.add_sample(HotelBookingSample(
                booking_id=sconf["booking_id"],
                check_in_date=sconf["check_in_date"],
                check_out_date=sconf["check_out_date"],
                num_guests=AtomicRangeInt(sconf["num_guests"], 1, 20),
                total_price=AtomicRangeFloat(sconf["total_price"], 0.0, 5000.0)
            ))
    else:
        print(f"Unrecognized dataset '{dataset_name}' in JSON pipeline.")
        sys.exit(1)

    evaluator = Evaluator()
    metrics = evaluator.evaluate_and_write_json(ds, output_json=eval_json_out)
    print("\n=== Pipeline Evaluate Results ===")
    print(metrics)

    if eval_json_out:
        print(f"Saved evaluation results to {eval_json_out}")

    # Only search_query has generation in this example
    if dataset_name == "search_query":
        descriptor = DatasetDescriptor()
        serialized_ds = descriptor.serialize(ds)
        generator = SampleGenerator()
        new_sample_json = generator.generate_new_sample(serialized_ds)
        print("\n=== Generated New Sample (Search) ===")
        print(new_sample_json)
    else:
        print("\n(No generation logic for this dataset in 'pipeline' command.)")
