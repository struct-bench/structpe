# -*- coding: utf-8 -*-
"""
llm_handler.py

Provides helper functions to access LLMs. Detects Azure configuration automatically.

**Execution Logic:**
1.  Checks for environment variables: ENDPOINT_URL, API_VERSION, DEPLOYMENT_NAME.
2.  **If ALL THREE are set:** Uses Azure OpenAI with Azure AD authentication via SDK.
    - Uses the original global client setup.
    - Requires `openai>=1.0.0`, `azure-identity`.
3.  **Otherwise (if any Azure var is missing):** Falls back to using LiteLLM.
    - Requires LITELLM_MODEL env var (e.g., "gemini/gemini-pro").
    - Requires corresponding provider API key env var (e.g., GEMINI_API_KEY).
    - Requires `litellm`.

Configuration:
 - **For Azure AD:** SET ENDPOINT_URL, API_VERSION, DEPLOYMENT_NAME. Ensure AD auth works.
 - **For LiteLLM:** UNSET at least ONE of ENDPOINT_URL, API_VERSION, DEPLOYMENT_NAME.
                   SET LITELLM_MODEL and the relevant API key.

Original file structure and function logic preserved, only call_gpt modified
for conditional execution.
"""

import os
import re
import json
import importlib  # For dynamic dataset importing

# --- Determine Execution Path based on Azure Vars ---
# Read the specific variables needed to trigger the Azure SDK path
AZURE_ENDPOINT_URL = os.getenv("ENDPOINT_URL")
AZURE_API_VERSION = os.getenv("API_VERSION")
AZURE_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME") # Use the original variable

# Determine if the Azure SDK path should be used
# This flag controls which logic path is taken in call_gpt
USE_AZURE_SDK_PATH = bool(AZURE_ENDPOINT_URL and AZURE_API_VERSION and AZURE_DEPLOYMENT_NAME)

# --- Conditional Imports and Client Setup ---
azure_client = None
litellm = None

if USE_AZURE_SDK_PATH:
    print(f"[llm_handler] Info: Azure ENV vars detected (Endpoint/Version/Deployment='{AZURE_DEPLOYMENT_NAME}'). Initializing Azure SDK path.")
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AzureOpenAI

        # Acquire token provider (as in original code)
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )

        # Create AzureOpenAI client globally (as in original code)
        # Ensure variables are not None before creating client
        if AZURE_ENDPOINT_URL and AZURE_API_VERSION:
             azure_client = AzureOpenAI(
                 azure_endpoint=AZURE_ENDPOINT_URL,
                 azure_ad_token_provider=token_provider,
                 api_version=AZURE_API_VERSION
             )
             print("[llm_handler] Info: AzureOpenAI client initialized successfully.")
        else:
             # Should not happen if USE_AZURE_SDK_PATH is True, but defensive check
             print("[llm_handler] Error: Azure config vars missing despite path detection.")
             raise ValueError("Internal Error: Azure config vars missing during client init.")

    except ImportError:
        print("[llm_handler] Error: Azure SDK libraries ('openai>=1.0.0', 'azure-identity') not found. "
              "Cannot use Azure AD path. Please install them.")
        # Set flag to False if imports fail, forcing fallback attempt (which might also fail)
        USE_AZURE_SDK_PATH = False
        # raise ImportError("Missing Azure SDK libraries required for detected Azure configuration.") # Option: Fail hard
    except Exception as e:
        print(f"[llm_handler] Error: Failed to initialize Azure client: {e}")
        USE_AZURE_SDK_PATH = False
        # raise RuntimeError(f"Failed to initialize Azure client: {e}") # Option: Fail hard

if not USE_AZURE_SDK_PATH:
    print("[llm_handler] Info: Azure ENV vars not fully set or Azure init failed. Attempting LiteLLM path.")
    try:
        import litellm
        print("[llm_handler] Info: LiteLLM library imported successfully.")
    except ImportError:
        print("[llm_handler] Error: LiteLLM library not found. Install using 'pip install litellm'. Cannot use LiteLLM path.")
        # litellm remains None, calls will fail later if this path is needed

# --- Original Utility Function ---
def remove_code_fences(text: str) -> str:
    """
    Remove triple-backtick code fences from GPT output.
    Helps avoid accidentally including Markdown blocks.
    """
    # Preserve original regex
    return re.sub(r"```[^`]*```", "", text)

# --- Modified Core LLM Call Function ---
def call_gpt(messages, temperature=0.7, top_p=0.9, max_tokens=800):
    """
    Generic call to the configured LLM. Auto-routes to Azure SDK if
    ENDPOINT_URL, API_VERSION, DEPLOYMENT_NAME are set, otherwise uses LiteLLM
    (requires LITELLM_MODEL and API key).
    """
    if USE_AZURE_SDK_PATH:
        # --- Azure SDK Path ---
        if azure_client is None:
             # This indicates an initialization failure happened earlier
             raise RuntimeError("Azure client was not initialized successfully. Check logs.")
        if not AZURE_DEPLOYMENT_NAME:
             # Should be set if USE_AZURE_SDK_PATH is True
             raise ValueError("Internal Error: Azure DEPLOYMENT_NAME is not set for Azure path.")

        #print(f"[llm_handler] Info: Calling Azure OpenAI deployment '{AZURE_DEPLOYMENT_NAME}' via SDK.")
        try:
            resp = azure_client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME, # Use deployment name from original env var
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            # Use original response extraction method
            rdict = resp.to_dict()
            content = rdict["choices"][0]["message"]["content"].strip()
            #print("[llm_handler] Info: Received response from Azure OpenAI.")
            return content
        except Exception as e:
            print(f"[llm_handler] Error: Error calling Azure OpenAI deployment '{AZURE_DEPLOYMENT_NAME}': {e}")
            raise e # Re-raise

    else:
        # --- LiteLLM Path ---
        if litellm is None:
            raise ImportError("LiteLLM library is required for this path but was not found.")

        LITELLM_MODEL = os.getenv("LITELLM_MODEL")
        if not LITELLM_MODEL:
            raise ValueError("LiteLLM path selected (Azure vars not set), but LITELLM_MODEL environment variable is missing.")

        print(f"[llm_handler] Info: Calling LiteLLM model '{LITELLM_MODEL}'.")
        try:
            resp = litellm.completion(
                model=LITELLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            # Use standard LiteLLM/OpenAI V1+ response extraction
            content = resp.choices[0].message.content.strip()
            print(f"[llm_handler] Info: Received response from LiteLLM model {LITELLM_MODEL}.")
            return content
        except Exception as e:
            print(f"[llm_handler] Error: Error calling LiteLLM model '{LITELLM_MODEL}': {e}")
            raise e # Re-raise


###############################################################################
# Helper functions to load dataset info - Kept EXACTLY as original
###############################################################################

def _get_dataset_module(dataset_name):
    """
    Dynamically import the dataset module for the given dataset_name.
    Expects a file named structpe/dataset/{dataset_name}_dataset.py
    """
    # Changed line: # <-- This comment was in your original, keeping it
    module_path = f"structpe.dataset.{dataset_name}_dataset"
    return importlib.import_module(module_path)

def _get_sample_attributes(dataset_name, sample):
    """
    Retrieve a text block describing this sample's attributes
    from the dataset's module (so the code is not hardcoded).
    The dataset module is expected to provide a function:
      build_sample_attribute_text(sample) -> str
    """
    dataset_module = _get_dataset_module(dataset_name)
    if hasattr(dataset_module, "build_sample_attribute_text"):
        return dataset_module.build_sample_attribute_text(sample)
    else:
        # Fallback if the dataset module doesn't implement a custom function
        # or if user wants a simpler default approach:
        # We just show the sample dict as a fallback
        return str(sample.__dict__)

def _get_graph_data(dataset_name, sample):
    """
    Retrieve the graph data from the dataset module, if available.
    The dataset module is expected to provide a function:
      get_graph_data(sample) -> str or dict
    """
    dataset_module = _get_dataset_module(dataset_name)
    if hasattr(dataset_module, "get_graph_data"):
        return dataset_module.get_graph_data(sample)
    else:
        # fallback: just read sample.sample_graph if it exists
        return getattr(sample, "sample_graph", None)

###############################################################################
# OLD FUNCTIONS (unchanged) - Kept EXACTLY as original
###############################################################################

def llm_judge_sample_prompt(dataset_name, sample):
    """
    Build a prompt that instructs the LLM to rate the correctness of a sample's attributes
    in [0..1].
    """
    attributes_text = _get_sample_attributes(dataset_name, sample)
    content = f"""
You are an AI that evaluates the correctness of a synthetic sample for dataset '{dataset_name}'.
The sample has these attributes:
{attributes_text}

Rate the overall correctness of these attributes from 0..1,
where 0=completely incorrect, 1=fully correct.
Output ONLY the float number.
"""
    return [{"role": "user", "content": content.strip()}]

def llm_judge_sample(dataset_name, sample) -> float:
    """
    Main simpler function to do a final float rating in [0..1].
    We'll parse the returned text as a float, fallback to 0.0 on error.
    """
    msgs = llm_judge_sample_prompt(dataset_name, sample)
    raw = call_gpt(msgs, temperature=0.0, max_tokens=100) # Calls modified call_gpt
    raw = remove_code_fences(raw) # Uses original remove_code_fences
    # Attempt parse (original logic)
    try:
        val = float(raw)
        if val < 0:
            val = 0.0
        if val > 1:
            val = 1.0
        return round(val, 3)
    except:
        return 0.0

def llm_judge_sample_with_debug(dataset_name, sample):
    """
    Returns a tuple: (score_float, prompt_messages, raw_response_text).
    1) build the prompt via llm_judge_sample_prompt
    2) call_gpt => raw text
    3) parse the text as float in [0..1], fallback 0.0
    4) return the numeric score, the entire 'messages' used, and the raw response
    """
    msgs = llm_judge_sample_prompt(dataset_name, sample)
    raw = call_gpt(msgs, temperature=0.0, max_tokens=100) # Calls modified call_gpt
    cleaned = remove_code_fences(raw) # Uses original remove_code_fences

    # parse the float (original logic)
    try:
        val = float(cleaned)
        if val < 0:
            val = 0.0
        if val > 1:
            val = 1.0
        score = round(val, 3)
    except:
        score = 0.0

    return (score, msgs, raw)  # raw is the uncleaned version if you want

def llm_check_graph(sample) -> (bool, str):
    """
    OLD function returning (bool, str). We'll keep it unchanged for backward compatibility.
    Checks if there's a 'sample_graph' and tries to see if it's DAG (OK) or not (FAIL).
    """
    if not hasattr(sample, "sample_graph"):
        return True, "No sample_graph found"

    # Original prompt building
    content = f"""
We have a sample with a 'sample_graph' adjacency dict: {sample.sample_graph}
Check if it forms a valid directed acyclic graph with no cycles
and makes sense for attribute generation order.
Reply with either:
  "OK" if it's valid
  "FAIL" if it's not
Then a short explanation.
"""
    msgs = [{"role":"user","content":content.strip()}]
    raw = call_gpt(msgs, temperature=0.2, max_tokens=100) # Calls modified call_gpt
    cleaned = remove_code_fences(raw).lower() # Uses original remove_code_fences
    # Original checking logic
    if "ok" in cleaned and "fail" not in cleaned:
        return (True, f"LLM says it's valid (raw='{raw}')")
    else:
        return (False, f"LLM might say it's invalid: {raw}")

###############################################################################
# NEW FUNCTIONS - Kept EXACTLY as original
###############################################################################

def llm_judge_graph_prompt(dataset_name, sample):
    """
    Build a prompt that instructs the LLM to rate how well the sample's 'sample_graph'
    constraints are satisfied, from 1..5 (1=very poor, 5=excellent).
    """
    graph_data = _get_graph_data(dataset_name, sample) # Uses original _get_graph_data
    content = f"""
You are an AI that evaluates how well a sample's 'sample_graph' constraints are satisfied
for dataset '{dataset_name}'.

'sample_graph': {graph_data}

Please provide a rating in [1..5], where:
  1 => completely violates adjacency constraints
  5 => fully satisfies adjacency constraints
Output ONLY the numeric rating. Then on a new line, a short explanation.
"""
    return [{"role":"user","content":content.strip()}]

def _parse_rating_1to5(raw_text: str) -> float:
    """
    Extract a numeric rating from the LLM response, clamped to [1..5].
    For example, if the raw_text is "4\nIt mostly works but has minor errors.",
    we parse '4' as float(4.0). If it's out of range, clamp to [1..5].
    """
    # Original parsing logic
    match = re.search(r"[-+]?\d*\.?\d+", raw_text)
    if not match:
        return 3.0  # fallback

    val_str = match.group(0)
    try:
        val = float(val_str)
        if val < 1:
            val = 1.0
        if val > 5:
            val = 5.0
        return round(val, 2)
    except:
        return 3.0

def llm_judge_graph(dataset_name, sample) -> float:
    """
    Non-debug function that returns a float in [1..5].
    We'll parse the returned text with _parse_rating_1to5().
    """
    msgs = llm_judge_graph_prompt(dataset_name, sample)
    raw = call_gpt(msgs, temperature=0.0, max_tokens=150) # Calls modified call_gpt
    cleaned = remove_code_fences(raw) # Uses original remove_code_fences
    score = _parse_rating_1to5(cleaned) # Uses original _parse_rating_1to5
    return score

def llm_judge_graph_with_debug(dataset_name, sample):
    """
    Debug version returning (score_in_1to5, messages_used, raw_response).
    """
    msgs = llm_judge_graph_prompt(dataset_name, sample)
    raw = call_gpt(msgs, temperature=0.0, max_tokens=150) # Calls modified call_gpt
    cleaned = remove_code_fences(raw) # Uses original remove_code_fences
    score = _parse_rating_1to5(cleaned) # Uses original _parse_rating_1to5
    return (score, msgs, raw)