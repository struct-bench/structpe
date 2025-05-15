"""
evaluator.py: The Evaluator class that runs sample verifications, distribution analysis, 
and can write detailed stats to a JSON file, including average stay length for hotel bookings, etc.
"""

import json
import datetime
from collections import Counter
from structpe.evaluator.evaluator_types import LLMJudge, Verifier, Metrics

class Evaluator:
    """
    Evaluates a dataset by:
     - Checking each sample's constraints (re-calling sample.verify() in a try/except)
     - Using LLMJudge for a random 'score'
     - Using Verifier for text presence
     - Aggregating distribution of e.g. intent, sentiment, or recommendation
     - For a HotelBooking dataset, computing average price/stay length

    If output_json is specified, results are written to that file.
    """
    def __init__(self):
        self.judge = LLMJudge()
        self.verifier = Verifier()

    def evaluate_and_write_json(self, dataset_obj, output_json=None):
        """
        Main public method to evaluate the dataset and optionally write stats to a JSON file.
        """
        stats = self._evaluate_dataset(dataset_obj)

        # If user requested writing to a JSON file, do so here:
        if output_json:
            with open(output_json, "w") as f:
                json.dump(stats, f, indent=2)

        return stats

    def _evaluate_dataset(self, dataset_obj):
        """
        Internal method that returns a dictionary of evaluation results. 
        Also demonstrates specialized stats for 'hotel_booking' constraints.
        """
        samples = getattr(dataset_obj, "samples", [])
        if not samples:
            return {
                "total_samples": 0,
                "valid_count": 0,
                "invalid_count": 0,
                "invalid_samples": [],
                "distribution": {},
                "sample_level_results": []
            }

        total_samples = len(samples)
        valid_count = 0
        invalid_count = 0
        invalid_samples = []

        # We'll track the total random LLMJudge score to compute an average
        total_score = 0.0

        # We'll store sample-level results (score, is_valid) in a list
        sample_results = []

        # We'll store distribution by intent, sentiment, or recommendation
        distribution_counter = Counter()

        # Additional custom stats for a hotel booking dataset
        sum_prices = 0.0
        sum_stays = 0
        booking_count = 0

        for sample in samples:
            # Attempt to re-verify constraints on the sample
            try:
                sample.verify()
                is_invalid = False
            except Exception as e:
                # If it fails, we record it as invalid
                is_invalid = True
                invalid_count += 1
                invalid_samples.append({
                    "sample": str(sample),  # or maybe sample.booking_id if available
                    "error": str(e)
                })

            # LLMJudge random score
            score = self.judge.judge_sample(sample)
            total_score += score

            # Verifier check (just checks if there's some non-empty text, or else)
            is_valid_text = self.verifier.verify_sample(sample)

            # If it's not invalid from constraints, and the text is valid, we consider overall valid
            # (This is a simplification: real logic might differ.)
            if (not is_invalid) and is_valid_text:
                valid_count += 1

            # Check distribution
            if hasattr(sample, "intent"):
                distribution_counter[sample.intent.value] += 1
            elif hasattr(sample, "sentiment"):
                distribution_counter[sample.sentiment.value] += 1
            elif hasattr(sample, "recommendation"):
                distribution_counter[sample.recommendation.value] += 1

            # If it's a HotelBookingSample, let's parse out stay length for extra stats
            if hasattr(sample, "booking_id") and hasattr(sample, "check_in_date") and hasattr(sample, "check_out_date"):
                try:
                    check_in_dt = datetime.datetime.strptime(sample.check_in_date, "%Y-%m-%d").date()
                    check_out_dt = datetime.datetime.strptime(sample.check_out_date, "%Y-%m-%d").date()
                    stay_len = (check_out_dt - check_in_dt).days
                    sum_stays += stay_len

                    # total_price is an AtomicRangeFloat
                    sum_prices += sample.total_price.value
                    booking_count += 1
                except:
                    pass

            # Gather sample-level metrics
            sample_results.append(Metrics(score, not is_invalid).to_dict())

        # Compute averages
        avg_score = round(total_score / total_samples, 3)
        avg_price = 0.0
        avg_stay = 0.0
        if booking_count > 0:
            avg_price = round(sum_prices / booking_count, 2)
            avg_stay = round(sum_stays / booking_count, 2)

        # Build final stats dictionary
        stats = {
            "total_samples": total_samples,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "invalid_samples": invalid_samples,
            "average_llm_score": avg_score,
            "distribution": dict(distribution_counter),
            "sample_level_results": sample_results
        }

        # If we found hotel bookings, add extra stats
        if booking_count > 0:
            stats["average_price"] = avg_price
            stats["average_stay_length"] = avg_stay

        return stats
