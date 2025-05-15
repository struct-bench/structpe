import json
import os
import importlib  # <-- ADDED to handle dynamic import
from collections import Counter
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- NEW CODE ---
from scipy.stats import wasserstein_distance
# --- end of NEW CODE ---

# Import from evaluator_types (no circular imports!)
from structpe.evaluator.evaluator_types import (
    LLMJudge, # needs LLM
    Verifier,
    Metrics,
    GraphEvaluator, # needs LLM
    GenericGrammarCheck,
    DatasetLevelVerifier,
    KNNMetrics,
    DatasetDependentMetrics  # handles dataset_metric_*
)

# NEW: Import Tregex evaluator
from structpe.evaluator.evaluator_types import TregexQueryEvaluator

class Evaluator:
    """
    Evaluates each sample, does optional dataset-level verification,
    and saves results to JSON + plots.

    If 'original_dataset' is provided, we use it for embedding references and also
    auto-attach it to dataset_module as 'private_dataset' so we can compute distribution distances.
    """

    def __init__(self, dataset_name="unknown", original_dataset=None, dataset_module=None):
        """
        :param dataset_name: The string name (e.g. "sentiment")
        :param original_dataset: The original dataset used for embeddings, etc.
        :param dataset_module: Optional pre-imported dataset module
        """
        self.dataset_name = dataset_name
        self.original_dataset = original_dataset
        self.dataset_module = dataset_module  # If provided, store it now.

        self.llm_judge = None
        self.verifier = None
        self.metrics = None
        self.graph_evaluator = None
        self.grammar_checker = None
        self.dataset_level_verifier = None
        self.knn_metrics = None
        self.dataset_metrics = None

        # 1) LLMJudge
        if os.environ.get('ENDPOINT_URL') is not None:             
            try:
                self.llm_judge = LLMJudge(dataset_name=dataset_name)
            except Exception as e:
                print(f"[Evaluator] WARNING: LLMJudge init => {e}")

        # 2) Verifier => pass/fail
        try:
            self.verifier = Verifier(dataset_module=self.dataset_module)
        except Exception as e:
            print(f"[Evaluator] WARNING: Verifier init => {e}")
            self.verifier = None

        # 3) Metrics => embedding similarity
        try:
            self.metrics = Metrics(original_dataset)
        except Exception as e:
            print(f"[Evaluator] WARNING: Metrics init => {e}")
            self.metrics = None

        # 4) GraphEvaluator => adjacency correctness in [1..5]
        if os.environ.get('ENDPOINT_URL') is not None: 
            try:
                self.graph_evaluator = GraphEvaluator(dataset_name=dataset_name)
            except Exception as e:
                print(f"[Evaluator] WARNING: GraphEvaluator init => {e}")
                self.graph_evaluator = None

        # 5) GrammarCheck => optional parse
        try:
            self.grammar_checker = GenericGrammarCheck(dataset_name)
        except Exception as e:
            print(f"[Evaluator] WARNING: GrammarCheck init => {e}")
            self.grammar_checker = None

        # 6) Dataset-level verifier => optional global checks
        try:
            self.dataset_level_verifier = DatasetLevelVerifier(dataset_module=self.dataset_module)
        except Exception as e:
            print(f"[Evaluator] WARNING: DatasetLevelVerifier init => {e}")
            self.dataset_level_verifier = None

        # 7) KNN-based metrics
        try:
            self.knn_metrics = KNNMetrics(original_dataset=original_dataset, k=3)
        except Exception as e:
            print(f"[Evaluator] WARNING: KNNMetrics init => {e}")
            self.knn_metrics = None

        # 8) Tregex-based evaluator
        try:
            self.tregex_evaluator = TregexQueryEvaluator()
        except Exception as e:
            print(f"[Evaluator] WARNING: TregexQueryEvaluator init => {e}")
            self.tregex_evaluator = None

        # 9) Attempt to attach private metrics
        try:
            from structpe.evaluator.evaluator_types import PrivateDatasetReferenceMetrics
            self.private_ref_metrics = PrivateDatasetReferenceMetrics(dataset_module=self.dataset_module)
        except Exception as e:
            print(f"[Evaluator] WARNING: PrivateDatasetReferenceMetrics init => {e}")
            self.private_ref_metrics = None

        # Track # of LLM calls
        self.num_llm_calls = 0

    def evaluate_full(self, dataset_obj, dataset_module=None, eval_json_out=None, savedir=None, plot=False):
        # If user provides a new dataset_module => override
        if dataset_module:
            self.dataset_module = dataset_module
            self.dataset_metrics = DatasetDependentMetrics(dataset_module)
            print("[Evaluator] Using user-provided dataset_module for metrics.")
        else:
            # If we don't already have one, auto-import it
            if not self.dataset_module:
                mod_path = f"structpe.dataset.{self.dataset_name}_dataset"
                try:
                    auto_mod = importlib.import_module(mod_path)
                    self.dataset_module = auto_mod
                    self.dataset_metrics = DatasetDependentMetrics(auto_mod)
                    print(f"[Evaluator] Auto-imported dataset module => {mod_path}")
                except Exception as e:
                    print(f"[Evaluator] WARNING: Could not auto-import => {e}")
                    self.dataset_module = None
                    self.dataset_metrics = None
            else:
                if not self.dataset_metrics and self.dataset_module:
                    self.dataset_metrics = DatasetDependentMetrics(self.dataset_module)

        # If we have an original_dataset => attach as private
        if self.original_dataset and self.dataset_module:
            print("[Evaluator] Auto-attaching original_dataset as 'private_dataset' in dataset_module.")
            setattr(self.dataset_module, "private_dataset", self.original_dataset)

        # Make directory if needed
        if savedir:
            os.makedirs(savedir, exist_ok=True)
            results_dir = savedir
        else:
            results_dir = "save_dir"
            os.makedirs(results_dir, exist_ok=True)

        # If no eval_json_out => default
        if not eval_json_out:
            eval_json_out = os.path.join(results_dir, "eval_results.json")

        # Make sure verifiers have the final dataset_module
        if self.verifier:
            self.verifier.dataset_module = self.dataset_module
        if self.dataset_level_verifier:
            self.dataset_level_verifier.dataset_module = self.dataset_module

        # Evaluate dataset
        results_dict = self._evaluate_dataset(dataset_obj)

        # Insert # of LLM calls
        if "aggregate" in results_dict and "global" in results_dict["aggregate"]:
            results_dict["aggregate"]["global"]["llm_api_calls_made"] = self.num_llm_calls

        # If dataset-level verifier => aggregator
        if self.dataset_level_verifier:
            dataset_debug = self.dataset_level_verifier.to_debug_dict(dataset_obj)
            results_dict["aggregate"]["dataset_verifier"] = {
                "used": dataset_debug["dataset_verify_used"],
                "passed_count": dataset_debug["passed_count"],
                "failed_count": dataset_debug["failed_count"],
                "failures": dataset_debug["failures"]
            }

        # Clean raw_sample_obj
        for sid, sdata in results_dict.get("samples", {}).items():
            raw_obj = sdata.get("raw_sample_obj", None)
            if raw_obj:
                sdata["original_sample_data"] = {
                    "text": getattr(raw_obj, "text", None),
                    "sentiment": str(getattr(raw_obj, "sentiment", None)),
                    "emotion": str(getattr(raw_obj, "emotion", None)),
                    "rating": getattr(raw_obj, "rating", None)
                }
                del sdata["raw_sample_obj"]

        # Convert float32 => float
        results_dict = self._convert_float32_to_float(results_dict)

        # Write final JSON
        with open(eval_json_out, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)
        print(f"[Evaluator] Wrote evaluation results to '{eval_json_out}'")

        # Generate plots if requested
        if plot:
            self._generate_plots(results_dict, results_dir)

        # Print final JSON summary, then the table
        print("\n=== EVALUATION RESULTS (JSON) ===\n")
        print(json.dumps(results_dict, indent=2))
        
        # Print the new aggregator table
        print("\n=== AGGREGATION STATS TABLE ===\n")
        table_str = self._build_final_metrics_table(results_dict)
        print(table_str)

        return results_dict

    def _evaluate_dataset(self, dataset_obj):
        samples = getattr(dataset_obj, "samples", [])
        total = len(samples)
        if total == 0:
            return self._empty_result()

        verifier_passed = 0
        verifier_failed = 0
        grammar_pass_count = 0
        sum_llm = 0.0
        sum_sim = 0.0
        sum_graph = 0.0

        correctness_scores = []
        similarities = []
        graph_scores = []
        synthetic_texts = []
        samples_results = {}

        # Possibly define node_pairs
        node_pairs = []
        if self.dataset_module and hasattr(self.dataset_module,"compute_node_similarities"):
            maybe_val = getattr(self.dataset_module, "compute_node_similarities")
            if callable(maybe_val):
                node_pairs = maybe_val()
            elif isinstance(maybe_val, list):
                node_pairs = maybe_val

        # Possibly define Tregex queries
        tregex_queries = []
        if self.dataset_module and hasattr(self.dataset_module, "tregex_queries"):
            tregex_queries = getattr(self.dataset_module, "tregex_queries", [])

        # Evaluate each sample
        for idx, sample in enumerate(tqdm(samples, desc="Evaluating samples", unit="sample")):
            exception_msg = None
            try:
                if hasattr(sample, "verify") and callable(sample.verify):
                    sample.verify()
            except Exception as e:
                exception_msg = str(e)

            # 1) LLM correctness
            correctness_sc = 0.0
            if self.llm_judge:
                try:
                    correctness_sc = self.llm_judge.judge_sample(sample)
                    self.num_llm_calls += 1
                except Exception as e:
                    correctness_sc = 0.0
                    print(f"[Evaluator] WARNING: LLMJudge => {e}")
            sum_llm += correctness_sc
            correctness_scores.append(correctness_sc)

            # 2) Verifier => pass/fail
            text_ok = True
            if self.verifier:
                try:
                    text_ok = self.verifier.verify_sample(sample)
                except Exception as e:
                    text_ok = False
                    print(f"[Evaluator] WARNING: verifier => {e}")
            final_ok = (exception_msg is None) and text_ok
            if final_ok:
                verifier_passed += 1
            else:
                verifier_failed += 1

            # 3) Embedding similarity
            sim_val = 0.0
            if self.metrics and self.metrics.model:
                try:
                    sim_val = self.metrics.compare(sample)
                except Exception as e:
                    sim_val = 0.0
                    print(f"[Evaluator] WARNING: metrics => {e}")
            sum_sim += sim_val
            similarities.append(sim_val)

            # 4) Graph adjacency => [1..5]
            graph_sc = 0.0
            graph_msg = "No response"
            if self.graph_evaluator:
                try:
                    sc, msg = self.graph_evaluator.evaluate_graph(sample)
                    graph_sc = sc
                    graph_msg = msg
                    self.num_llm_calls += 1
                except Exception as e:
                    graph_sc = 0.0
                    graph_msg = f"Graph => {e}"
                    print(f"[Evaluator] WARNING: graph => {e}")
            sum_graph += graph_sc
            graph_scores.append(graph_sc)

            # 5) Grammar => parse from grammar checker nodes
            grammar_ok = False
            grammar_trace = {}
            node_similarity_matrix = []
            parsed_nodes = []
            pairwise_scores = []

            if self.grammar_checker:
                try:
                    # This returns (bool, nodes), but we only store the bool. Next line fetches debug (inc. nodes).
                    grammar_ok = self.grammar_checker.check_sample(sample)
                    grammar_trace = self.grammar_checker.to_debug_dict(sample)

                    # Reuse the same node list from the grammar trace
                    parsed_nodes = grammar_trace.get("nodes", [])

                    # If grammar is OK & we have embeddings => compute NxN adjacency
                    if grammar_ok and self.metrics and self.metrics.model and parsed_nodes:
                        grammar_pass_count += 1
                        # Combine each node's key:value into one text line for embedding
                        lines = [f"{nd['field']}:{nd['value']}" for nd in parsed_nodes]
                        emb = self.metrics.model.encode(lines, convert_to_numpy=True)
                        mat = cosine_similarity(emb)
                        node_similarity_matrix = mat.astype(float).tolist()

                        n = len(lines)
                        # NxN adjacency => store pairwise
                        for i2 in range(n):
                            for j2 in range(i2+1, n):
                                sc_ij = float(mat[i2][j2])
                                pairwise_scores.append({
                                    "nodeA": i2,
                                    "nodeB": j2,
                                    "similarity": round(sc_ij,8)
                                })
                except Exception as e:
                    print(f"[Evaluator] WARNING: grammar => {e}")

            # Collect text for KNN
            syn_text = getattr(sample, "text", "")
            synthetic_texts.append(syn_text)

            # 6) Tregex evaluation
            tregex_res = {}
            if self.tregex_evaluator and tregex_queries:
                grammar_module_path = f"structpe.dataset.{self.dataset_name}_dataset"
                try:
                    # Use the same grammar string as the checker
                    tregex_res = self.tregex_evaluator.run_tregex(
                        text=self.grammar_checker.build_grammar_string(sample),
                        grammar_module_path=grammar_module_path,
                        queries=tregex_queries
                    )
                except Exception as e:
                    tregex_res = {"error": str(e), "queries_results": []}

            # Build final sample record
            sample_record = {
                "verifier": {
                    "text_check_passed": text_ok,
                    "exception_during_verify": exception_msg,
                    "trace_debug": self._safe_debug_dict(self.verifier, sample)
                },
                "llm_judge": {
                    "correctness_score": correctness_sc,
                    "trace_debug": self._safe_judge_debug_dict(sample)
                },
                "metrics": {
                    "embedding_similarity": sim_val,
                    "trace_debug": self._safe_metrics_debug_dict(sample)
                },
                "graph_evaluator": {
                    "graph_score": graph_sc,
                    "graph_message": graph_msg,
                    "trace_debug": self._safe_graph_debug_dict(sample)
                },
                "grammar_check": {
                    "passed": grammar_ok,
                    "trace_debug": grammar_trace,
                    "nodes": parsed_nodes,           # the same node list from grammar_trace
                    "pairwise_scores": pairwise_scores,
                    "full_matrix": node_similarity_matrix
                },
                "tregex_evaluator": tregex_res,
                "raw_sample_obj": sample
            }

            # If grammar_ok & node_pairs => compute node pair similarities
            if grammar_ok and node_pairs and parsed_nodes and node_similarity_matrix:
                field2idx = {}
                for nd in parsed_nodes:
                    # 'index' is a string, so cast to int for indexing
                    field2idx[nd["field"]] = int(nd["index"])
                n = len(parsed_nodes)
                for (left_field, right_field) in node_pairs:
                    key_name = f"{left_field}-{right_field}-similarity"
                    pair_val = None
                    if left_field in field2idx and right_field in field2idx:
                        iA = field2idx[left_field]
                        iB = field2idx[right_field]
                        if 0 <= iA < n and 0 <= iB < n:
                            pair_val = node_similarity_matrix[iA][iB]
                    sample_record["grammar_check"][key_name] = round(pair_val, 8) if pair_val else None

            samples_results[str(idx)] = sample_record

        # aggregator
        avg_llm = round(sum_llm / total, 8)
        avg_sim = round(sum_sim / total, 8)
        avg_graph = round(sum_graph / total, 8)

        graph_agg = {
            "average_score": avg_graph,
            "scores_list": [round(s,8) for s in graph_scores],
            "min_score": round(min(graph_scores), 8) if graph_scores else 0.0,
            "max_score": round(max(graph_scores), 8) if graph_scores else 0.0
        }

        # KNN metrics
        knn_out = {}
        if self.knn_metrics:
            try:
                comp = self.knn_metrics.compute_knn_metrics(synthetic_texts)
                for i, neighbor_list in comp["sample_neighbors"].items():
                    samples_results[str(i)]["knn_neighbors"] = neighbor_list
                knn_out = {
                    "knn_precision": comp["precision"],
                    "knn_recall": comp["recall"],
                    "unique_neighbors_count": comp["unique_neighbors_count"]
                }
            except Exception as e:
                print(f"[Evaluator] WARNING: KNN => {e}")
                knn_out = {
                    "knn_precision": 0.0,
                    "knn_recall": 0.0,
                    "unique_neighbors_count": 0
                }

        results_dict = {
            "samples": samples_results,
            "aggregate": {
                "verifier": {
                    "passed_count": verifier_passed,
                    "failed_count": (total - verifier_passed)
                },
                "llm_judge": {
                    "average_score": avg_llm
                },
                "metrics": {
                    "average_similarity": avg_sim
                },
                "graph_evaluator": graph_agg,
                "grammar_check": {
                    "passed_count": grammar_pass_count,
                    "failed_count": (total - grammar_pass_count)
                },
                "knn_metrics": knn_out,
                "dataset_dependent_metrics": {},
                "global": {
                    "total_samples": total,
                    "llm_api_calls_made": 0
                }
            }
        }

        # If dataset_metrics => run them
        if self.dataset_metrics:
            try:
                metric_out = self.dataset_metrics.evaluate(dataset_obj, samples_results)
                results_dict["aggregate"]["dataset_dependent_metrics"] = metric_out
            except Exception as e:
                results_dict["aggregate"]["dataset_dependent_metrics"] = {
                    "error": f"Could not run dataset_metric_* => {e}"
                }
        else:
            results_dict["aggregate"]["dataset_dependent_metrics"] = {
                "debug_info": "No dataset_module or no dataset_metric_* found."
            }

        # Node-sim aggregator from grammar_check
        node_sim_data = {}
        for sid, rec in samples_results.items():
            grammar_check = rec.get("grammar_check", {})
            for k, val in grammar_check.items():
                if k.endswith("-similarity"):
                    if k not in node_sim_data:
                        node_sim_data[k] = []
                    node_sim_data[k].append(val)

        node_sim_agg = {}
        for key, arr in node_sim_data.items():
            valid_vals = [x for x in arr if x is not None]
            none_count = len(arr) - len(valid_vals)
            avg_val = round(sum(valid_vals)/len(valid_vals),8) if valid_vals else None
            node_sim_agg[key] = {
                "average": avg_val,
                "valid_count": len(valid_vals),
                "none_count": none_count,
                # store the actual distribution of synthetic data
                "all_values_synthetic": valid_vals  
            }
        results_dict["aggregate"]["grammar_check"]["node_similarity_aggregate"] = node_sim_agg

        # Summarize sample-level metrics in dataset_dependent_metrics
        if "dataset_dependent_metrics" in results_dict["aggregate"]:
            ddm = results_dict["aggregate"]["dataset_dependent_metrics"]
            sample_metrics_summary = {}
            for sid, rec in samples_results.items():
                if "sample_dependent_metrics" in rec:
                    for metric_name, metric_res in rec["sample_dependent_metrics"].items():
                        if metric_name not in sample_metrics_summary:
                            sample_metrics_summary[metric_name] = {
                                "values": [],
                                "none_count": 0,
                                "error_count": 0
                            }
                        if metric_res is None:
                            sample_metrics_summary[metric_name]["none_count"] += 1
                            sample_metrics_summary[metric_name]["values"].append(metric_res)
                        elif isinstance(metric_res, str) and metric_res.startswith("ERROR =>"):
                            sample_metrics_summary[metric_name]["error_count"] += 1
                            sample_metrics_summary[metric_name]["values"].append(metric_res)
                        else:
                            sample_metrics_summary[metric_name]["values"].append(metric_res)
            for m_name, aggregator_data in sample_metrics_summary.items():
                aggregator_data["total_count"] = len(aggregator_data["values"])
                aggregator_data["debug_info"] = (
                    f"Sample-level metric '{m_name}': total={aggregator_data['total_count']}, "
                    f"none={aggregator_data['none_count']}, errors={aggregator_data['error_count']}"
                )
            ddm["sample_level_summary"] = sample_metrics_summary

        # --- Evaluate private_dataset if present ---
        private_dataset_results = {}
        private_distributions = {}  # --- NEW FOR PER-PAIR DIST ---
        if self.dataset_module and hasattr(self.dataset_module, "private_dataset"):
            private_ds = getattr(self.dataset_module, "private_dataset", None)
            if private_ds and hasattr(private_ds, "samples"):
                p_samples = getattr(private_ds, "samples", [])
                pass_count_private = 0
                private_samples_dict = {}

                # For each node-pair => we store an array of private values
                for (left_field, right_field) in node_pairs:
                    key_name = f"{left_field}-{right_field}-similarity"
                    private_distributions[key_name] = []

                for p_idx, p_sample in enumerate(tqdm(p_samples, desc="Private dataset grammar", unit="sample")):
                    grammar_ok = False
                    grammar_trace = {}
                    node_similarity_matrix = []
                    node_pairs_data = {}

                    if self.grammar_checker:
                        try:
                            grammar_ok = self.grammar_checker.check_sample(p_sample)
                            grammar_trace = self.grammar_checker.to_debug_dict(p_sample)
                            if grammar_ok and self.metrics and self.metrics.model:
                                pass_count_private += 1
                                parsed_nodes = grammar_trace.get("nodes", [])
                                lines = [f"{nd['field']}:{nd['value']}" for nd in parsed_nodes]

                                if lines:
                                    emb = self.metrics.model.encode(lines, convert_to_numpy=True)
                                    mat = cosine_similarity(emb)

                                    # parse lines => field->index
                                    field2idx = {}
                                    for line_i, line_str in enumerate(lines):
                                        field_part = line_str
                                        if ":" in line_str:
                                            parts = line_str.split(":", 1)
                                            field_part = parts[0].strip()
                                        field2idx[field_part] = line_i

                                    # compute node pair similarities
                                    for (left_field, right_field) in node_pairs:
                                        key_name = f"{left_field}-{right_field}-similarity"
                                        pair_val = None
                                        if left_field in field2idx and right_field in field2idx:
                                            iA = field2idx[left_field]
                                            iB = field2idx[right_field]
                                            if 0 <= iA < len(lines) and 0 <= iB < len(lines):
                                                pair_val = mat[iA][iB]
                                        # store
                                        node_pairs_data[key_name] = round(pair_val,8) if pair_val else None
                                        if pair_val is not None:
                                            private_distributions[key_name].append(pair_val)
                        except Exception as e:
                            print(f"[Evaluator] WARNING: private dataset grammar => {e}")

                    p_rec = {
                        "grammar_check_passed": grammar_ok,
                        "trace_debug": grammar_trace,
                        "node_pairs_data": node_pairs_data
                    }
                    private_samples_dict[str(p_idx)] = p_rec

                private_dataset_results = {
                    "passed_count": pass_count_private,
                    "failed_count": len(p_samples) - pass_count_private,
                    # we store the distributions of all pairs in "node_similarity_distributions_private" for clarity
                    "node_similarity_distributions_private": private_distributions,
                    "samples": private_samples_dict
                }

        # place private dataset info in final JSON
        results_dict["private_dataset_parsing"] = private_dataset_results

        # --- NEW FOR PER-PAIR DIST ---
        # We'll compute a dictionary of pair_key => distance, and store the synthetic vs private arrays
        pairwise_distances = {}
        node_sim_agg = results_dict["aggregate"]["grammar_check"].get("node_similarity_aggregate", {})

        # We'll also store them in results_dict["aggregate"]["knn_distribution_distance"] as a dict
        pairwise_dist_dict = {}
        for key_name, main_agg in node_sim_agg.items():
            main_vals = main_agg.get("all_values_synthetic", [])
            priv_vals = private_distributions.get(key_name, [])
            # store them so user can see
            pairwise_dist_dict[key_name] = {
                "synthetic_scores": main_vals,
                "private_scores": priv_vals,
                "wasserstein_distance": 0.0
            }
            if main_vals and priv_vals:
                dist_val = wasserstein_distance(main_vals, priv_vals)
                pairwise_dist_dict[key_name]["wasserstein_distance"] = round(dist_val,8)

        # Then store pairwise_dist_dict as "knn_distribution_distance" => a dictionary
        results_dict["aggregate"]["knn_distribution_distance"] = pairwise_dist_dict

        return results_dict

    def _safe_debug_dict(self, verifier_obj, sample):
        if not verifier_obj:
            return {}
        try:
            return verifier_obj.to_debug_dict(sample)
        except Exception as e:
            print(f"[Evaluator] WARNING: to_debug_dict => {e}")
            return {"error": str(e)}

    def _safe_judge_debug_dict(self, sample):
        if not self.llm_judge:
            return {}
        try:
            sc, dbg = self.llm_judge.judge_sample_debug(sample)
            return dbg
        except Exception as e:
            print(f"[Evaluator] WARNING: llm_judge debug => {e}")
            return {"error": str(e)}

    def _safe_metrics_debug_dict(self, sample):
        if not self.metrics:
            return {}
        try:
            return self.metrics.to_debug_dict(sample)
        except Exception as e:
            print(f"[Evaluator] WARNING: metrics debug => {e}")
            return {"error": str(e)}

    def _safe_graph_debug_dict(self, sample):
        if not self.graph_evaluator:
            return {}
        try:
            return self.graph_evaluator.to_debug_dict(sample)
        except Exception as e:
            print(f"[Evaluator] WARNING: graph debug => {e}")
            return {"error": str(e)}

    def _convert_float32_to_float(self, data):
        if isinstance(data, dict):
            return {k: self._convert_float32_to_float(v) for k,v in data.items()}
        elif isinstance(data, list):
            return [self._convert_float32_to_float(x) for x in data]
        elif isinstance(data, float):
            return data
        elif isinstance(data, np.floating):
            return float(data)
        else:
            if data is not None and not isinstance(data, (int,str,bool,float)):
                print(f"[Evaluator] WARNING: Found unrecognized type '{type(data)}' => converting to string.")
            return str(data)

    def _generate_plots(self, results_dict, results_dir):
        aggregator = results_dict.get("aggregate", {})
        plot_data = results_dict.get("_plot_data", {})

        correctness_scores = plot_data.get("correctness_scores", [])
        similarities = plot_data.get("similarities", [])
        graph_scores = plot_data.get("graph_scores", [])

        # correctness
        if correctness_scores:
            plt.figure(figsize=(6,4))
            plt.hist(correctness_scores, bins=5, range=(1,6), edgecolor='black')
            plt.title("LLM Correctness Scores Distribution")
            plt.xlabel("Score (1..5)")
            plt.ylabel("Count")
            plt.tight_layout()
            outpath = os.path.join(results_dir, "hist_correctness_scores.png")
            plt.savefig(outpath)
            plt.close()
            print(f"[Evaluator] Saved correctness score histogram to {outpath}")

        # embedding similarity
        if similarities:
            plt.figure(figsize=(6,4))
            plt.hist(similarities, bins=10, range=(0,1), edgecolor='black')
            plt.title("Embedding Similarities Distribution")
            plt.xlabel("Similarity (0..1)")
            plt.ylabel("Count")
            plt.tight_layout()
            outpath = os.path.join(results_dir, "hist_embedding_similarity.png")
            plt.savefig(outpath)
            plt.close()
            print(f"[Evaluator] Saved embedding similarity histogram to {outpath}")

        # graph scores
        if graph_scores:
            plt.figure(figsize=(6,4))
            plt.hist(graph_scores, bins=5, range=(1,6), edgecolor='black')
            plt.title("Graph Score Distribution")
            plt.xlabel("Score (1..5)")
            plt.ylabel("Count")
            plt.tight_layout()
            outpath = os.path.join(results_dir, "hist_graph_scores.png")
            plt.savefig(outpath)
            plt.close()
            print(f"[Evaluator] Saved graph score histogram to {outpath}")

        # grammar pass/fail
        grammar_agg = aggregator.get("grammar_check", {})
        grammar_pass = grammar_agg.get("passed_count", 0)
        grammar_fail = grammar_agg.get("failed_count", 0)
        plt.figure(figsize=(5,4))
        plt.bar(["Pass", "Fail"], [grammar_pass, grammar_fail], color=["green","red"])
        plt.title("Grammar Check Results")
        plt.xlabel("Outcome")
        plt.ylabel("Count")
        plt.tight_layout()
        outpath = os.path.join(results_dir, "bar_grammar_check.png")
        plt.savefig(outpath)
        plt.close()
        print(f"[Evaluator] Saved grammar check bar chart to {outpath}")

        # verifier pass/fail
        verifier_agg = aggregator.get("verifier", {})
        vpass = verifier_agg.get("passed_count", 0)
        vfail = verifier_agg.get("failed_count", 0)
        plt.figure(figsize=(5,4))
        plt.bar(["Pass", "Fail"], [vpass, vfail], color=["green","red"])
        plt.title("Verifier Pass/Fail")
        plt.xlabel("Outcome")
        plt.ylabel("Count")
        plt.tight_layout()
        outpath = os.path.join(results_dir, "bar_verifier_check.png")
        plt.savefig(outpath)
        plt.close()
        print(f"[Evaluator] Saved verifier pass/fail bar chart to {outpath}")

    def _empty_result(self):
        return {
            "samples": {},
            "aggregate": {
                "verifier": {"passed_count": 0, "failed_count": 0},
                "llm_judge": {"average_score": 0.0},
                "metrics": {"average_similarity": 0.0},
                "graph_evaluator": {"average_score": 0.0},
                "grammar_check": {"passed_count": 0, "failed_count": 0},
                "knn_metrics": {"knn_precision": 0.0, "knn_recall": 0.0},
                "dataset_dependent_metrics": {},
                "global": {"total_samples": 0, "llm_api_calls_made": 0}
            }
        }

    # ─────────────────────────────────────────────────────────
    # NEW HELPER: Build a readable ASCII table of final metrics
    # ─────────────────────────────────────────────────────────
    def _build_final_metrics_table(self, results_dict):
        aggregator = results_dict.get("aggregate", {})
        global_info = aggregator.get("global", {})
        grammar_info = aggregator.get("grammar_check", {})

        # For safety, cast them all to int
        total_synthetic = int(global_info.get("total_samples", 0))
        grammar_pass    = int(grammar_info.get("passed_count", 0))

        private_parsing = results_dict.get("private_dataset_parsing", {})
        total_private   = len(private_parsing.get("samples", {}))

        # Then the pass rate
        cfg_pass_rate = 0.0
        if total_synthetic > 0:
            cfg_pass_rate = grammar_pass / total_synthetic

        # 4) Node similarities in aggregate
        node_sim_agg = grammar_info.get("node_similarity_aggregate", {})

        # 5) KNN distribution distance
        knn_distances = aggregator.get("knn_distribution_distance", {})

        # 6) KNN precision
        knn_info = aggregator.get("knn_metrics", {})
        knn_precision = knn_info.get("knn_precision", 0.0)

        # 7) KNN recall
        knn_recall = knn_info.get("knn_recall", 0.0)

        # Build table lines
        lines = []
        lines.append("-----------------------------------------------------")
        lines.append(f"Total Synthetic Samples : {total_synthetic}")
        lines.append(f"Total Private Samples   : {total_private}")
        lines.append(f"CFG Pass Rate          : {cfg_pass_rate:.3f}  ({grammar_pass}/{total_synthetic})")

        lines.append("")
        lines.append("Node Similarities (Average per pair):")
        if node_sim_agg:
            for pair_key, data in node_sim_agg.items():
                avg_val = data.get("average", None)
                lines.append(f"   {pair_key}: avg={avg_val}, valid_count={data.get('valid_count')}, none_count={data.get('none_count')}")
        else:
            lines.append("   (No node-sim pairs found)")

        lines.append("")
        lines.append("KNN Distribution Distances (Wasserstein):")
        if knn_distances:
            for dist_key, dist_info in knn_distances.items():
                wdist = dist_info.get("wasserstein_distance", 0.0)
                lines.append(f"   {dist_key}: {wdist}")
        else:
            lines.append("   (No distribution distances found)")

        lines.append("")
        lines.append(f"KNN Precision: {knn_precision:.3f}")
        lines.append(f"KNN Recall   : {knn_recall:.3f}")
        lines.append("-----------------------------------------------------")

        return "\n".join(lines)


def compare_eval_results(base_data, comp_data, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    base_agg = base_data.get("aggregate", {})
    comp_agg = comp_data.get("aggregate", {})

    import matplotlib.pyplot as plt

    # LLM correctness
    base_llm = base_agg.get("llm_judge", {}).get("average_score", 0)
    comp_llm = comp_agg.get("llm_judge", {}).get("average_score", 0)
    plt.figure(figsize=(5,4))
    plt.bar(["Base","Comp"], [base_llm, comp_llm], color=["blue","orange"])
    plt.title("Compare LLM Avg Correctness")
    plt.xlabel("Dataset")
    plt.ylabel("LLM Score [1..5]")
    plt.tight_layout()
    outp = os.path.join(out_dir, "compare_llm_correctness.png")
    plt.savefig(outp)
    plt.close()
    print(f"[CompareEval] Saved LLM correctness comparison to {outp}")

    # Verifier pass/fail
    b_vpass = base_agg.get("verifier", {}).get("passed_count", 0)
    b_vfail = base_agg.get("verifier", {}).get("failed_count", 0)
    c_vpass = comp_agg.get("verifier", {}).get("passed_count", 0)
    c_vfail = comp_agg.get("verifier", {}).get("failed_count", 0)

    x_labels = ["Base", "Comp"]
    pass_vals = [b_vpass, c_vpass]
    fail_vals = [b_vfail, c_vfail]

    plt.figure(figsize=(6,4))
    x_positions = [0,1]
    plt.bar(x_positions, pass_vals, color="green", label="Pass")
    plt.bar(x_positions, fail_vals, bottom=pass_vals, color="red", label="Fail")
    plt.xticks(x_positions, x_labels)
    plt.title("Verifier Comparison (Pass/Fail)")
    plt.xlabel("Dataset")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    outp = os.path.join(out_dir, "compare_verifier.png")
    plt.savefig(outp)
    plt.close()
    print(f"[CompareEval] Saved verifier comparison to {outp}")

    # embedding similarity
    b_sim = base_agg.get("metrics", {}).get("average_similarity", 0.0)
    c_sim = comp_agg.get("metrics", {}).get("average_similarity", 0.0)
    plt.figure(figsize=(5,4))
    plt.bar(["Base","Comp"], [b_sim, c_sim], color=["blue","orange"])
    plt.title("Compare Average Embedding Similarity")
    plt.xlabel("Dataset")
    plt.ylabel("Similarity (0..1)")
    plt.tight_layout()
    outp = os.path.join(out_dir, "compare_similarity.png")
    plt.savefig(outp)
    plt.close()
    print(f"[CompareEval] Saved embedding similarity comparison to {outp}")

    print("[CompareEval] Done generating comparison plots.")
