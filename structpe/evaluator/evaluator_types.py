import os
import random
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from pdb import set_trace as bp

from structpe.utilities import llm_handler

##########################################################
#   DATASET-LEVEL VERIFIER
##########################################################
class DatasetLevelVerifier:
    def __init__(self, dataset_module=None):
        self.dataset_module = dataset_module
        print(f"[DatasetLevelVerifier] Initialized with dataset_module={dataset_module}")

    def verify_dataset(self, dataset_obj) -> list:
        """
        If dataset_module has verify_dataset(...), call it. Return list of (idx, bool, reason).
        """
        if not self.dataset_module:
            print("[DatasetLevelVerifier] No dataset_module => skipping dataset-level verify.")
            return []
        fn = getattr(self.dataset_module, "verify_dataset", None)
        if not callable(fn):
            print("[DatasetLevelVerifier] No verify_dataset function found => skipping.")
            return []
        try:
            print("[DatasetLevelVerifier] Calling dataset_module.verify_dataset(...)")
            out = fn(dataset_obj)
            print(f"[DatasetLevelVerifier] verify_dataset returned {len(out)} checks.")
            return out
        except Exception as e:
            print(f"[DatasetLevelVerifier] WARNING: verify_dataset call failed => {e}")
            return []

    def to_debug_dict(self, dataset_obj) -> dict:
        res = self.verify_dataset(dataset_obj)
        if not res:
            return {
                "dataset_verify_used": False,
                "failures": [],
                "passed_count": 0,
                "failed_count": 0,
                "debug_info": "No dataset-level verification or no checks performed."
            }
        failures = []
        passed_count = 0
        failed_count = 0
        for (idx, pass_bool, reason) in res:
            if pass_bool:
                passed_count += 1
            else:
                failed_count += 1
                failures.append((idx, reason))
        return {
            "dataset_verify_used": True,
            "failures": failures,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "debug_info": f"Dataset-level checks => total={len(res)}, pass={passed_count}, fail={failed_count}"
        }


##########################################################
#   LLMJudge
##########################################################
class LLMJudge:
    def __init__(self, dataset_name="unknown"):
        self.dataset_name = dataset_name
        print(f"[LLMJudge] Initialized for dataset='{dataset_name}'.")

    def judge_sample(self, sample) -> float:
        try:
            return llm_handler.llm_judge_sample(self.dataset_name, sample)
        except Exception as e:
            print(f"[LLMJudge] WARNING: llm_judge_sample failed => {e}")
            return 0.0

    def judge_sample_debug(self, sample):
        """
        Returns (score, debug_dict).
        """
        try:
            sc, msgs, raw_resp = llm_handler.llm_judge_sample_with_debug(
                self.dataset_name, sample
            )
            return sc, {
                "llm_prompt_messages": msgs,
                "llm_raw_response": raw_resp,
                "final_score": sc
            }
        except Exception as e:
            print(f"[LLMJudge] WARNING: judge_sample_debug => {e}")
            return 0.0, {
                "llm_prompt_messages": [],
                "llm_raw_response": "ERROR",
                "final_score": 0.0
            }

    def to_debug_dict(self, sample) -> dict:
        sc, dbg = self.judge_sample_debug(sample)
        return {
            "judge_method": "LLM-based scoring in [1..5]",
            "dataset_name": self.dataset_name,
            "score": round(sc, 2),
            "llm_prompt_messages": dbg.get("llm_prompt_messages", []),
            "llm_raw_response": dbg.get("llm_raw_response", ""),
            "debug_info": f"LLMJudge => final_score={sc}"
        }


##########################################################
#   Verifier - sample-level
##########################################################
class Verifier:
    def __init__(self, dataset_module=None):
        self.dataset_module = dataset_module
        print(f"[Verifier] Initialized with dataset_module={dataset_module}")

    def verify_sample(self, sample) -> bool:
        """
        If dataset_module has verify_sample(sample), call it. Otherwise fallback => text presence.
        """
        try:
            if (self.dataset_module and 
                hasattr(self.dataset_module, "verify_sample") and
                callable(self.dataset_module.verify_sample)):
                return self.dataset_module.verify_sample(sample)
            else:
                txt = getattr(sample, "text", None)
                return bool(txt and txt.strip())
        except Exception as e:
            print(f"[Verifier] WARNING: verify_sample => {e}")
            return False

    def to_debug_dict(self, sample) -> dict:
        passed = self.verify_sample(sample)
        used_custom = (
            self.dataset_module
            and hasattr(self.dataset_module, "verify_sample")
            and callable(self.dataset_module.verify_sample)
        )
        return {
            "verification_method": "custom_dataset" if used_custom else "fallback_text_presence",
            "verification_passed": passed,
            "debug_info": f"Verifier => used_custom={used_custom}, final_passed={passed}"
        }


##########################################################
#   Metrics - embedding similarity
##########################################################
class Metrics:
    """
    Build embeddings from the *private/original* dataset. Then compare each *synthetic* sample text to them.
    """

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.original_texts = []
        print("[Metrics] Initializing text embeddings...")

        if original_dataset and hasattr(original_dataset, "samples"):
            try:
                for s in original_dataset.samples:
                    txt = getattr(s, "text", None)
                    if txt: 
                        self.original_texts.append(txt)
            except Exception as e:
                print(f"[Metrics] WARNING: reading original samples => {e}")
                self.original_texts = []

        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[Metrics] Loaded SentenceTransformer('all-MiniLM-L6-v2').")
        except Exception as e:
            print(f"[Metrics] WARNING: cannot init model => {e}")
            self.model = None

        self.orig_embeddings = None
        if self.original_texts and self.model:
            try:
                self.orig_embeddings = self.model.encode(self.original_texts, convert_to_numpy=True)
                print(f"[Metrics] Embedded {len(self.original_texts)} original samples for similarity.")
            except Exception as e:
                print(f"[Metrics] WARNING: embedding original texts => {e}")
                self.orig_embeddings = None

    def compare(self, synthetic_sample):
        """
        For a given synthetic sample, measure max cos sim vs. all original_text embeddings.
        """
        if not self.model or self.orig_embeddings is None or self.orig_embeddings.size == 0:
            return 0.0
        try:
            syn_text = getattr(synthetic_sample, "text", "")
            if not syn_text:
                return 0.0
            syn_emb = self.model.encode([syn_text], convert_to_numpy=True)
            sims = cosine_similarity(syn_emb, self.orig_embeddings)
            return round(float(np.max(sims)), 3)
        except Exception as e:
            print(f"[Metrics] WARNING: compare => {e}")
            return 0.0

    def to_debug_dict(self, synthetic_sample):
        val = self.compare(synthetic_sample)
        return {
            "similarity_method": "max_cosine_similarity",
            "model_used": "all-MiniLM-L6-v2",
            "final_similarity": val,
            "debug_info": f"Compared synthetic sample text to {len(self.original_texts)} private texts."
        }


##########################################################
#   GraphLLMJudge + GraphEvaluator
##########################################################
class GraphLLMJudge:
    def __init__(self, dataset_name="unknown"):
        self.dataset_name = dataset_name
        print(f"[GraphLLMJudge] Initialized for dataset='{dataset_name}'.")

    def judge_graph(self, sample) -> float:
        try:
            return llm_handler.llm_judge_graph(self.dataset_name, sample)
        except Exception as e:
            print(f"[GraphLLMJudge] WARNING: judge_graph => {e}")
            return 0.0

    def judge_graph_debug(self, sample):
        try:
            sc, msgs, raw = llm_handler.llm_judge_graph_with_debug(self.dataset_name, sample)
            return sc, {
                "llm_prompt_messages": msgs,
                "llm_raw_response": raw,
                "final_score": sc
            }
        except Exception as e:
            print(f"[GraphLLMJudge] WARNING: judge_graph_debug => {e}")
            return 0.0, {
                "llm_prompt_messages": [],
                "llm_raw_response": "ERROR",
                "final_score": 0.0
            }

    def to_debug_dict(self, sample):
        sc, dbg = self.judge_graph_debug(sample)
        return {
            "graph_llm_judge_method": "LLM-based adjacency correctness in [1..5]",
            "dataset_name": self.dataset_name,
            "score": round(sc, 2),
            "llm_prompt_messages": dbg.get("llm_prompt_messages", []),
            "llm_raw_response": dbg.get("llm_raw_response", ""),
            "debug_info": f"GraphLLMJudge => final_score={sc}"
        }


class GraphEvaluator:
    def __init__(self, dataset_name="unknown"):
        try:
            self.llm_judge = GraphLLMJudge(dataset_name=dataset_name)
        except Exception as e:
            print(f"[GraphEvaluator] WARNING: GraphLLMJudge init => {e}")
            self.llm_judge = None
        print(f"[GraphEvaluator] Initialized => dataset_name={dataset_name}")

    def evaluate_graph(self, sample):
        if not self.llm_judge:
            return 0.0, "No graph judge available"
        try:
            sc, dbg_info = self.llm_judge.judge_graph_debug(sample)
            return sc, dbg_info.get("llm_raw_response", "No response")
        except Exception as e:
            print(f"[GraphEvaluator] WARNING: evaluate_graph => {e}")
            return 0.0, f"GraphEvaluator error => {e}"

    def to_debug_dict(self, sample):
        if not self.llm_judge:
            return {
                "graph_llm_judge_method": "missing",
                "score": 0.0,
                "llm_prompt_messages": [],
                "llm_raw_response": "No graph judge",
                "debug_info": "GraphEvaluator => no llm_judge"
            }
        try:
            sc, dbg = self.llm_judge.judge_graph_debug(sample)
            return {
                "graph_llm_judge_method": "LLM-based adjacency correctness in [1..5]",
                "score": round(sc, 2),
                "llm_prompt_messages": dbg.get("llm_prompt_messages", []),
                "llm_raw_response": dbg.get("llm_raw_response", ""),
                "debug_info": f"GraphEvaluator => final_score={sc}"
            }
        except Exception as e:
            print(f"[GraphEvaluator] WARNING: to_debug_dict => {e}")
            return {
                "graph_llm_judge_method": "error",
                "score": 0.0,
                "llm_prompt_messages": [],
                "llm_raw_response": "GraphEvaluator to_debug_dict error",
                "debug_info": f"GraphEvaluator => exception={e}"
            }


##########################################
#   GENERIC GRAMMAR CHECK  –  fixed
##########################################
##########################################
#   GENERIC GRAMMAR CHECK – unify nodes
##########################################
import importlib, ast
from typing import List, Dict, Any

from lark import Token, Tree
from structpe.utilities.grammar_handling import (
    build_parser,
    check_sample_against_grammar,
)

class GenericGrammarCheck:
    """
    1) If grammar_string is a Python dict => parse with ast.literal_eval() => key→value nodes.
    2) Otherwise parse with Lark => for each 'pair' subtree (key ":" value),
       we gather (key, value).

    Debug prints are added so you can see the parse tree and confirm what's happening.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name   = dataset_name
        self.dataset_module = None
        self.parser         = None
        self.last_nodes: List[Dict[str, Any]] = []

        print(f"[GenericGrammarCheck] Initializing => dataset_name='{dataset_name}'")

        # 1) dynamic import of dataset module
        try:
            mod_path = f"structpe.dataset.{dataset_name}_dataset"
            self.dataset_module = importlib.import_module(mod_path)
            print(f"[GenericGrammarCheck] Imported module={mod_path}")
        except Exception as e:
            print(f"[GenericGrammarCheck] WARNING: import dataset module => {e}")
            return

        # 2) fetch grammar
        grammar_str = None
        try:
            if hasattr(self.dataset_module, "SAMPLE_GRAMMAR"):
                grammar_str = getattr(self.dataset_module, "SAMPLE_GRAMMAR")
            elif hasattr(self.dataset_module, "get_grammar"):
                grammar_str = self.dataset_module.get_grammar()
        except Exception as e:
            print(f"[GenericGrammarCheck] WARNING: retrieving grammar => {e}")

        if not grammar_str:
            print("[GenericGrammarCheck] WARNING: no grammar => skipping checks.")
            return

        print(f"[GenericGrammarCheck] DEBUG: Grammar (SAMPLE_GRAMMAR) =>\n{grammar_str}")

        # 3) build parser
        try:
            self.parser = build_parser(grammar_str)
            print("[GenericGrammarCheck] Parser built successfully.")
        except Exception as e:
            print(f"[GenericGrammarCheck] WARNING: building parser => {e}")

    # ─────────────────────────────────────────
    # High-level build_grammar_string
    # ─────────────────────────────────────────
    def build_grammar_string(self, sample) -> str:
        if not self.dataset_module:
            return str(sample.__dict__)
        fn = getattr(self.dataset_module, "build_grammar_string_for_check", None)
        if callable(fn):
            try:
                return fn(sample)
            except Exception as e:
                print(f"[GenericGrammarCheck] WARNING: build_grammar_string_for_check => {e}")
        return str(sample.__dict__)

    # ─────────────────────────────────────────
    # Step 1: Try literal_eval if Python-like
    # ─────────────────────────────────────────
    def _extract_nodes_fast(self, s: str) -> List[Dict[str, Any]]:
        """
        If string is a Python dict, parse with ast.literal_eval -> gather items -> node list.
        """
        try:
            data = ast.literal_eval(s)
            if isinstance(data, dict):
                return [
                    {"index": str(i), "field": k, "value": v}
                    for i, (k, v) in enumerate(data.items())
                ]
        except Exception as e:
            # debug
            #print(f"[GenericGrammarCheck] DEBUG: literal_eval fail => {e}")
            pass
        return []

    # ─────────────────────────────────────────
    # Step 2: If that fails, parse with Lark => walk subtrees
    # ─────────────────────────────────────────
    def _extract_nodes_lark(self, s: str) -> List[Dict[str, Any]]:
        if not self.parser:
            return []

        print(f"\n[GenericGrammarCheck] DEBUG: About to parse => {repr(s)}")

        try:
            tree = self.parser.parse(s)
            # Print the parse tree for debugging
            print("[GenericGrammarCheck] DEBUG: parse tree =>")
            print(tree.pretty())
        except Exception as e:
            print(f"[GenericGrammarCheck] WARNING: parse => {e}")
            return []

        # We do a DFS:
        # For each subtree with node.data == "pair", we expect children [key, ":", value].
        return self._extract_pairs(tree)

    def _extract_pairs(self, root: Tree) -> List[Dict[str, Any]]:
        results = []
        
        def get_token_string(node):
            # Recursively gather text from all token leaves
            if isinstance(node, Token):
                return node.value
            if not hasattr(node, "children"):
                return str(node)
            return "".join(get_token_string(child) for child in node.children)

        def walk(t):
            if isinstance(t, Tree) and t.data == "pair":
                # pair => [key_subtree, ":", value_subtree]
                key_subtree = t.children[0]
                val_subtree = t.children[2]
                field_str = get_token_string(key_subtree).strip('"\'')
                value_str = get_token_string(val_subtree).strip('"\'')
                results.append({
                    "index": str(len(results)),
                    "field": field_str,
                    "value": value_str
                })
            for child in t.children:
                if isinstance(child, Tree):
                    walk(child)

        walk(root)
        return results

    # ─────────────────────────────────────────
    # Master dispatcher
    # ─────────────────────────────────────────
    def _extract_nodes(self, s: str) -> List[Dict[str, Any]]:
        # 1) Try to interpret as Python dict
        nodes = self._extract_nodes_fast(s)
        if nodes:
            #print("[GenericGrammarCheck] DEBUG: literal_eval => returning nodes")
            return nodes

        # 2) Otherwise parse with Lark -> walk 'pair' subtrees
        out_nodes = self._extract_nodes_lark(s)
        #print(f"[GenericGrammarCheck] DEBUG: _extract_nodes => final out_nodes => {out_nodes}")
        return out_nodes

    # ─────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────
    def check_sample(self, sample) -> (bool, List[Dict[str, Any]]):
        s = self.build_grammar_string(sample)
        print(f"[GenericGrammarCheck] DEBUG: build_grammar_string => {repr(s)}")
        self.last_nodes = self._extract_nodes(s)
        if not self.parser:
            return False, self.last_nodes

        try:
            # Attempt grammar check with Lark
            passed = check_sample_against_grammar(self.parser, s)
        except Exception as e:
            print(f"[GenericGrammarCheck] WARNING: check_sample => {e}")
            passed = False

        return passed, self.last_nodes

    def to_debug_dict(self, sample) -> Dict[str, Any]:
        if not self.last_nodes:
            _ = self._extract_nodes(self.build_grammar_string(sample))
        passed, _ = self.check_sample(sample)
        return {
            "grammar_check_dataset":  self.dataset_name,
            "grammar_string":         self.build_grammar_string(sample),
            "grammar_check_passed":   str(passed),
            "debug_info":             f"GrammarCheck => parse success={passed}",
            "nodes":                  self.last_nodes,
        }

##########################################################
#   KNNMetrics
##########################################################
class KNNMetrics:
    """
    **KNN Metrics** with k=1..N for approximate retrieval.
    We do a two-way check: synthetic->original => "precision",
    original->synthetic => "recall".
    """
    def __init__(self, original_dataset, k=3, model_name='all-MiniLM-L6-v2'):
        self.k = k
        self.original_texts = []
        print(f"[KNNMetrics] Initializing => k={k}, model_name='{model_name}'")

        if original_dataset and hasattr(original_dataset, "samples"):
            for s in original_dataset.samples:
                # Check if 'text' exists and is not empty
                if hasattr(s, "text") and s.text is not None and s.text.strip():
                    self.original_texts.append(s.text)
                else:
                    # Fallback: serialize sample attributes into one string
                    serialized = " ".join(str(v) for k, v in s.__dict__.items() if v is not None)
                    self.original_texts.append(serialized)
        else:
            print("[KNNMetrics] WARNING: No original_dataset or no .samples => KNN won't index anything.")

        print(f"[KNNMetrics] Found {len(self.original_texts)} original texts for indexing.")

        try:
            self.model = SentenceTransformer(model_name)
            print("[KNNMetrics] Loaded SentenceTransformer for KNN search.")
        except Exception as e:
            print(f"[KNNMetrics] WARNING: cannot init model => {e}")
            self.model = None

        self.orig_embeddings = None
        self.index_orig = None
        if self.original_texts and self.model:
            try:
                emb = self.model.encode(self.original_texts, convert_to_numpy=True)
                d = emb.shape[1]
                self.index_orig = faiss.IndexFlatIP(d)
                faiss.normalize_L2(emb)
                self.index_orig.add(emb)
                self.orig_embeddings = emb
                print("[KNNMetrics] FAISS index built with original data (index_orig).")
            except Exception as e:
                print(f"[KNNMetrics] WARNING: building index => {e}")
                self.orig_embeddings = None
                self.index_orig = None

    def _embed_batch(self, texts):
        if not self.model:
            print("[KNNMetrics] No model => cannot embed texts.")
            return None
        emb = self.model.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(emb)
        return emb

    def compute_knn_metrics(self, synthetic_texts):
        if self.index_orig is None or (not synthetic_texts):
            print("[KNNMetrics] Skipping => no index_orig or no synthetic texts.")
            return {
                "sample_neighbors": {},
                "precision": 0.0,
                "recall": 0.0,
                "unique_neighbors_count": 0
            }

        syn_emb = self._embed_batch(synthetic_texts)
        if syn_emb is None:
            print("[KNNMetrics] No synthetic embeddings => returning empty KNN metrics.")
            return {
                "sample_neighbors": {},
                "precision": 0.0,
                "recall": 0.0,
                "unique_neighbors_count": 0
            }

        # Synthetic->Original
        distances, ids = self.index_orig.search(syn_emb, self.k)
        sample_neighbors = {}
        all_neighbors_orig = set()
        for i in range(len(synthetic_texts)):
            row_ids = ids[i]
            row_dist = distances[i]
            neighbors_info = []
            for j in range(self.k):
                nid = int(row_ids[j])
                scr = float(row_dist[j])
                neighbors_info.append((nid, scr))
                all_neighbors_orig.add(nid)
            sample_neighbors[i] = neighbors_info
        unique_neighbors_count = len(all_neighbors_orig)
        orig_count = len(self.original_texts)
        if orig_count == 0:
            return {
                "sample_neighbors": sample_neighbors,
                "precision": 0.0,
                "recall": 0.0,
                "unique_neighbors_count": 0
            }
        precision_val = round(unique_neighbors_count / orig_count, 8)

        # Original->Synthetic => build synthetic index
        d = syn_emb.shape[1]
        index_syn = faiss.IndexFlatIP(d)
        index_syn.add(syn_emb)
        distances2, ids2 = index_syn.search(self.orig_embeddings, self.k)

        all_neighbors_syn = set()
        for i in range(orig_count):
            row2_ids = ids2[i]
            for j in range(self.k):
                all_neighbors_syn.add(int(row2_ids[j]))

        synth_count = len(synthetic_texts)
        if synth_count == 0:
            recall_val = 0.0
        else:
            recall_val = round(len(all_neighbors_syn) / synth_count, 4)

        return {
            "sample_neighbors": sample_neighbors,
            "precision": precision_val,
            "recall": recall_val,
            "unique_neighbors_count": unique_neighbors_count
        }


##########################################################
#   DatasetDependentMetrics
##########################################################
class DatasetDependentMetrics:
    """
    We discover 'dataset_metric_*' callables in the dataset module.
    They can be dataset-level or sample-level, indicated by fn.dataset_metric_level = "dataset" or "sample".
    """

    def __init__(self, dataset_module=None):
        self.dataset_module = dataset_module
        self.dataset_level_funcs = []
        self.sample_level_funcs = []
        print(f"[DatasetDependentMetrics] Initializing => dataset_module={dataset_module}")

        if dataset_module:
            for attr_name in dir(dataset_module):
                if attr_name.startswith("dataset_metric_"):
                    fn = getattr(dataset_module, attr_name, None)
                    if callable(fn):
                        level = getattr(fn, "dataset_metric_level", "dataset") 
                        if level == "sample":
                            self.sample_level_funcs.append(fn)
                            print(f"[DatasetDependentMetrics] Found sample-level metric '{fn.__name__}'")
                        else:
                            self.dataset_level_funcs.append(fn)
                            print(f"[DatasetDependentMetrics] Found dataset-level metric '{fn.__name__}'")
        else:
            print("[DatasetDependentMetrics] WARNING: no dataset_module => no metric funcs discovered.")

    def evaluate_dataset_level(self, dataset_obj, all_samples_results):
        """
        Return { fn_name: result_dict or error_string }, for each dataset-level metric.
        """
        results = {}
        for fn in self.dataset_level_funcs:
            fn_name = fn.__name__
            try:
                print(f"[DatasetDependentMetrics] Running dataset-level metric => {fn_name}")
                out = fn(dataset_obj, all_samples_results)
                results[fn_name] = out
            except Exception as e:
                msg = f"ERROR => {fn_name}: {e}"
                results[fn_name] = msg
                print(f"[DatasetDependentMetrics] {msg}")
        return results

    def evaluate_sample_level(self, dataset_obj, all_samples_results):
        """
        For each function with level='sample', call for each sample => place in sample_dependent_metrics.
        """
        for fn in self.sample_level_funcs:
            fn_name = fn.__name__
            print(f"[DatasetDependentMetrics] Running sample-level metric => {fn_name}")
            for sid, sample_data in all_samples_results.items():
                sample_obj = sample_data.get("raw_sample_obj", None)
                if sample_obj is None:
                    continue
                if "sample_dependent_metrics" not in sample_data:
                    sample_data["sample_dependent_metrics"] = {}
                try:
                    out = fn(dataset_obj, {"raw_sample_obj": sample_obj})
                    sample_data["sample_dependent_metrics"][fn_name] = out
                except Exception as e:
                    msg = f"ERROR => {fn_name}: {e}"
                    sample_data["sample_dependent_metrics"][fn_name] = msg
                    print(f"[DatasetDependentMetrics] {msg}")

    def evaluate(self, dataset_obj, all_samples_results):
        if (not self.dataset_level_funcs) and (not self.sample_level_funcs):
            return {
                "debug_info": "No discovered dataset_metric_* (dataset or sample) in dataset_module."
            }

        dataset_results = self.evaluate_dataset_level(dataset_obj, all_samples_results)
        self.evaluate_sample_level(dataset_obj, all_samples_results)
        return {
            "dataset_metrics": dataset_results,
            "sample_metrics": "Populated in each sample's 'sample_dependent_metrics'"
        }


#######################################################################
# NEW: TregexQueryEvaluator
#######################################################################
from structpe.evaluator.tregex import parse_and_match_with_generic_grammar
# The parse_and_match function is the same as in your tregex.py script (or an import).
# We assume it is accessible here.

class TregexQueryEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"[TregexQueryEvaluator] Initializing => model_name='{model_name}'")
        try:
            self.model = SentenceTransformer(model_name)
            print("[TregexQueryEvaluator] Loaded SentenceTransformer successfully.")
        except Exception as e:
            print(f"[TregexQueryEvaluator] WARNING: cannot init model => {e}")
            self.model = None

    def run_tregex(self, text, grammar_module_path, queries):
        out = {"queries_results": []}

        for q in queries:
            # parse & match
            try:
                #print(text, grammar_module_path, q)
                captures = parse_and_match_with_generic_grammar(text, grammar_module_path, q)

                # -- NEW CHECK: make sure captures is a list
                if not isinstance(captures, list):
                    raise TypeError(
                        f"[TregexQueryEvaluator] parse_and_match_with_generic_grammar returned "
                        f"{type(captures)} instead of list for query '{q}'"
                    )

            except Exception as e:
                # If there's any error, store it in "error"
                print(f"[TregexQueryEvaluator] WARNING: query '{q}' => parse/match error => {e}")
                out["queries_results"].append({
                    "query": q,
                    "error": str(e),
                    "matches": []
                })
                continue

            # embed matched nodes
            matches_with_embeddings = []
            to_embed = []
            for matchdict in captures:
                # matchdict should be a dict
                node_txt = matchdict.get("node_text", "")
                to_embed.append(node_txt)

            # compute embeddings
            embeddings = None
            if self.model and to_embed:
                try:
                    embeddings = self.model.encode(to_embed, convert_to_numpy=True)
                except Exception as e:
                    # This happens too often
                    #print(f"[TregexQueryEvaluator] WARNING: embedding => {e}")
                    embeddings = None

            # store match results
            for i, matchdict in enumerate(captures):
                node_txt = matchdict.get("node_text", "")
                embedding_vec = None
                if embeddings is not None and i < len(embeddings):
                    embedding_vec = embeddings[i].tolist()

                row = {
                    "capture_name": matchdict.get("capture_name"),
                    "node_label": matchdict.get("node_label"),
                    "node_text": node_txt,
                    #"embedding": embedding_vec
                }
                matches_with_embeddings.append(row)

            # Additional pairwise similarity among matched nodes
            pairwise_sims = []
            if embeddings is not None and len(embeddings) > 1:
                mat = cosine_similarity(embeddings)
                mat = mat.astype(float).tolist()
                n = len(mat)
                for i2 in range(n):
                    for j2 in range(i2+1, n):
                        sc_ij = mat[i2][j2]
                        pairwise_sims.append({
                            "i": i2,
                            "j": j2,
                            "similarity": round(sc_ij,4)
                        })

            out["queries_results"].append({
                "query": q,
                "matches": matches_with_embeddings,
                "pairwise_node_sims": pairwise_sims
            })
        return out

