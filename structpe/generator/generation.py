#!/usr/bin/env python3
"""
generation.py

A single-level private evolution pipeline that:
 - fetches random + variation prompts from the dataset's get_pe_random_api_prompt() / get_pe_variation_api_prompt().
 - loads the private data from file_path (JSON, CSV, or TSV).
 - uses Azure OpenAI to generate random + variation.
 - returns final synthetic texts for structpe to save.

Supports advanced arguments from run.py:
   sim_mode, k, l, sigma, temperature, top_p, max_tokens
"""

import os
import json
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# structpe dataset registry
from structpe.dataset.registry import get_dataset_class

# Azure/OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

# Make sure nltk is ready
nltk.download('punkt', quiet=True)

###############################################################################
# Embedding Model Setup
###############################################################################
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_tokenizer = None
_model = None

def load_embedding_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print("[generation.py] Loading embedding model once...")
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

def embed_text(texts):
    load_embedding_model()
    print(f"[generation.py] Embedding {len(texts)} texts...")
    inputs = _tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = _model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def calculate_similarity(S_pri, S_syn, sim_mode="avg"):
    """
    sim_mode in [avg, max, min].
    """
    if not S_syn:
        return 0.0
    E_pri = embed_text(S_pri)
    E_syn = embed_text(S_syn)
    sim_matrix = cosine_similarity(E_pri, E_syn)

    if sim_mode == "avg":
        sample_sims = np.mean(sim_matrix, axis=0)
    elif sim_mode == "max":
        sample_sims = np.max(sim_matrix, axis=0)
    elif sim_mode == "min":
        sample_sims = np.min(sim_matrix, axis=0)
    else:
        raise ValueError(f"Invalid sim_mode={sim_mode} in [avg, max, min].")

    sim_val = float(np.mean(sample_sims))
    print(f"[generation.py] => S_pri vs S_syn similarity({sim_mode})={sim_val:.4f}")
    return sim_val

def calculate_ttr(samples):
    """
    Type-token ratio => (#unique_tokens / #total_tokens).
    """
    if not samples:
        return 0.0
    all_tokens = []
    for s in samples:
        all_tokens.extend(word_tokenize(s))
    ratio = float(len(set(all_tokens))) / len(all_tokens) if all_tokens else 0.0
    print(f"[generation.py] => TTR={ratio:.4f} (type-token ratio).")
    return ratio

###############################################################################
# Azure OpenAI Setup
###############################################################################
def create_azure_openai_client(endpoint, deployment):
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-05-01-preview"
    )
    print(f"[generation.py] Created AzureOpenAI client with endpoint={endpoint}, deployment={deployment}")
    return client

def deployment_env_or(default_name):
    """Return DEPLOYMENT_NAME if set, else fallback to default_name."""
    env = os.getenv("DEPLOYMENT_NAME", None)
    return env if env else default_name

###############################################################################
# Chat APIs with temperature, top_p, max_tokens
###############################################################################
def random_api(client, random_prompt, n_samples, temperature, top_p, max_tokens):
    """
    Single request to Azure OpenAI => 'n_samples' completions => returns list[str].
    """
    print(f"[generation.py] [random_api] => calling chat.completions with n={n_samples}, temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    try:
        resp = client.chat.completions.create(
            model=deployment_env_or("gpt-4"),
            messages=[{"role": "user", "content": random_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n_samples
        ).to_dict()
        choices = resp.get("choices", [])
        texts = [c["message"]["content"].strip() for c in choices]
        print(f"[generation.py] [random_api] => got {len(texts)} random samples.")
        if texts:
            print(f"[generation.py] [random_api] => First sample:\n{texts[0]}\n-----")
        return texts
    except Exception as e:
        print(f"[random_api] ERROR: {e}")
        return ["RANDOM_ERROR"] * n_samples

def variation_api(client, variation_prompt, text, temperature, top_p, max_tokens):
    """
    Variation call => single new completion for 'text'.
    """
    try:
        prompt = f"{variation_prompt}\n\n{text}"
        resp = client.chat.completions.create(
            model=deployment_env_or("gpt-4"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        ).to_dict()
        new_text = resp["choices"][0]["message"]["content"].strip()
        return new_text
    except Exception as e:
        print(f"[variation_api] ERROR: {e}")
        return f"{text} [VAR_ERROR]"

def parallel_variation(client, variation_prompt, texts, concurrency, temperature, top_p, max_tokens):
    print(f"[generation.py] [parallel_variation] => concurrency={concurrency}, total={len(texts)} texts to vary.")
    results = [None]*len(texts)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        fut_map = {
            executor.submit(variation_api, client, variation_prompt, t, temperature, top_p, max_tokens): i
            for i, t in enumerate(texts)
        }
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = f"[VARIATION_ERROR]{e}"
            results[idx] = res
    return results

###############################################################################
# Load Private Data (file_path => JSON/CSV/TSV => list[str])
###############################################################################
def load_private_data(file_path, file_type):
    print(f"[generation.py] [load_private_data] => {file_type=} {file_path=}")
    if not os.path.isfile(file_path):
        print(f"[WARN] No private data at {file_path} => returning empty.")
        return []
    if file_type == "csv":
        with open(file_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return []
        # skip header => join columns
        data = [" ".join(row) for row in rows[1:] if row]
        return data
    elif file_type == "tsv":
        with open(file_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f, delimiter="\t"))
        if len(rows) < 2:
            return []
        data = [" ".join(row) for row in rows[1:] if row]
        return data
    else:
        # assume JSON
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError("JSON must be top-level list.")
        data = []
        for obj in items:
            if isinstance(obj, dict) and "text" in obj:
                data.append(obj["text"])
            else:
                data.append(json.dumps(obj))
        return data

###############################################################################
# Single-Level PE (with advanced arguments)
###############################################################################
def single_level_pe(client, random_prompt, variation_prompt,
                    S_pri, concurrency, init_count, iterations,
                    sim_mode, k, l, sigma,
                    temperature, top_p, max_tokens):
    """
    Minimal single-level approach:
      1) S_t => random_api(..., n=init_count)
      2) each iteration:
         - Variation pass: each sample => K expansions => rewrite => combine
         - Evaluate => pick top init_count
         - Expand by L => variation => next S_t
    Return final S_t

    We apply 'sigma' noise to the sample fitness. Then we pick top init_count.
    'sim_mode' can be 'avg','max','min'.
    'k' => how many expansions per sample
    'l' => how many times to replicate selected set
    We also pass 'temperature','top_p','max_tokens' to the LLM calls.
    """
    print("[generation.py] [single_level_pe] => Starting single-level PE pipeline.")
    print(f"[single_level_pe] => init_count={init_count}, iterations={iterations}, sim_mode={sim_mode}, K={k}, L={l}, sigma={sigma}")
    print(f"[single_level_pe] => LLM params => temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")

    # Step 1: S_t => random
    print(f"[single_level_pe] => Creating initial S_t with random_api(n={init_count}).")
    S_t = random_api(client, random_prompt, init_count, temperature, top_p, max_tokens)
    print(f"[single_level_pe] => initial S_t size={len(S_t)}")

    # Now proceed with iterations
    for it in range(iterations):
        print(f"\n[generation.py] SINGLE-LEVEL PE => ITERATION {it}")
        # Variation pass => each sample => K expansions
        new_batch = []
        for s in S_t:
            for _ in range(k):
                new_batch.append(s)
        print(f"[iteration={it}] => Variation pass from {len(S_t)} to {len(new_batch)} items. (K={k})")

        expanded = parallel_variation(client, variation_prompt, new_batch,
                                      concurrency, temperature, top_p, max_tokens)

        S_t_expanded = S_t + expanded
        print(f"[iteration={it}] => total candidates => {len(S_t_expanded)}")

        # Evaluate fitness => average sim => add noise => pick top init_count
        print(f"[iteration={it}] => compute sample_fitness => sim_mode={sim_mode}")
        E_pri = embed_text(S_pri)
        E_syn = embed_text(S_t_expanded)

        # sim_matrix => shape=(len(S_pri), len(S_t_expanded))
        # We will compute the mean across axis=0 for 'avg' if sim_mode='avg'.
        # (We can keep it simple and just re-use `calculate_similarity` in a loop, but let's do it in-place.)
        if sim_mode == "avg":
            sample_fitness = np.mean(np.dot(E_pri, E_syn.T), axis=0) / (np.linalg.norm(E_pri,axis=1)[:,None]*np.linalg.norm(E_syn,axis=1))
            # Actually we might do the same approach as your prior code to be consistent
            # We'll do => sim = np.mean(cosine_similarity(E_pri, E_syn), axis=0)
            sim_matrix = cosine_similarity(E_pri, E_syn)
            sample_fitness = np.mean(sim_matrix, axis=0)
        elif sim_mode == "max":
            sim_matrix = cosine_similarity(E_pri, E_syn)
            sample_fitness = np.max(sim_matrix, axis=0)
        elif sim_mode == "min":
            sim_matrix = cosine_similarity(E_pri, E_syn)
            sample_fitness = np.min(sim_matrix, axis=0)
        else:
            raise ValueError(f"Invalid sim_mode={sim_mode}")

        noise = np.random.normal(0, sigma, len(sample_fitness))
        sample_fitness_noisy = sample_fitness + noise

        # sort descending
        sorted_idx = np.argsort(sample_fitness_noisy)[::-1]
        top_idx = sorted_idx[:init_count]
        selected = [S_t_expanded[i] for i in top_idx]

        # Expand by L => variation => next S_t
        big_list = []
        for _ in range(l):
            big_list.extend(selected)

        S_t = parallel_variation(client, variation_prompt, big_list,
                                 concurrency, temperature, top_p, max_tokens)

        # debug => iteration-level similarity
        iter_sim = calculate_similarity(S_pri, S_t, sim_mode)
        print(f"[iteration={it}] => new S_t size={len(S_t)}, similarity({sim_mode})={iter_sim:.4f}")

    return S_t

###############################################################################
# run_generation_pipeline
###############################################################################
def run_generation_pipeline(file_path: str,
                            file_type: str = "json",
                            concurrency: int = 3,
                            init_count: int = 3,
                            iterations: int = 2,
                            endpoint: str = "https://syndata.openai.azure.com/",
                            deployment: str = "gpt-4",
                            dataset_name: str = None,
                            sim_mode: str = "avg",
                            k: int = 2,
                            l: int = 1,
                            sigma: float = 0.1,
                            temperature: float = 0.7,
                            top_p: float = 0.9,
                            max_tokens: int = 150):
    """
    The main function called by structpe generate:
      - We import the dataset class from dataset_name, to get random & variation prompts
      - We load the private data from file_path (json/csv/tsv)
      - We do single-level-pe
      - Return final list of strings

    Accepts advanced arguments:
      sim_mode, k, l, sigma, temperature, top_p, max_tokens

    Make sure your run.py calls them as well, e.g. run_generation_pipeline(
        file_path=...,
        file_type=...,
        concurrency=...,
        init_count=...,
        iterations=...,
        endpoint=...,
        deployment=...,
        dataset_name=...,
        sim_mode=args.sim_mode,
        k=args.k,
        l=args.l,
        sigma=args.sigma,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    """
    print(f"[run_generation_pipeline] => dataset_name={dataset_name}, file_path={file_path}, concurrency={concurrency}, init_count={init_count}, iterations={iterations}")
    print(f"[run_generation_pipeline] => sim_mode={sim_mode}, k={k}, l={l}, sigma={sigma}, temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")

    # 1) get dataset class => get the random & variation prompts
    ds_cls = get_dataset_class(dataset_name)
    ds_obj = ds_cls()  # e.g. GroundedQADataset() or SentimentDataset() etc.

    random_prompt = ds_obj.get_pe_random_api_prompt()
    variation_prompt = ds_obj.get_pe_variation_api_prompt()
    print(f"[run_generation_pipeline] => Fetched random_prompt and variation_prompt from '{dataset_name}' dataset class.")

    # 2) create azure openai client
    client = create_azure_openai_client(endpoint, deployment)

    # 3) load private data
    S_pri = load_private_data(file_path, file_type)
    print(f"[run_generation_pipeline] => loaded {len(S_pri)} private samples from {file_path}.")
    if S_pri:
        # show part of the first sample
        print(f"[run_generation_pipeline] => First private sample:\n  {S_pri[0][:200]}...\n")

    # 4) single-level-pe with advanced arguments
    final_syn = single_level_pe(
        client=client,
        random_prompt=random_prompt,
        variation_prompt=variation_prompt,
        S_pri=S_pri,
        concurrency=concurrency,
        init_count=init_count,
        iterations=iterations,
        sim_mode=sim_mode,
        k=k,
        l=l,
        sigma=sigma,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    # 5) summary
    print("[run_generation_pipeline] => Done single_level_pe. Calculating final similarity & TTR.")
    final_sim = calculate_similarity(S_pri, final_syn, sim_mode)
    final_ttr = calculate_ttr(final_syn)
    print(f"[run_generation_pipeline] => final #samples={len(final_syn)}, sim({sim_mode})={final_sim:.4f}, TTR={final_ttr:.4f}")

    return final_syn
