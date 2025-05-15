# test_llm_handler.py (Universal Version)
import unittest
import os
import json
import importlib
from unittest.mock import patch, MagicMock
import sys
import logging
import textwrap

# --- Configuration & Pre-Checks ---
print("--- Test Setup: Reading Configuration ---")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CRUCIAL: Read the target LLM model from the environment
LITELLM_MODEL_STRING = os.getenv("LITELLM_MODEL")
print(f"Read LITELLM_MODEL: {LITELLM_MODEL_STRING if LITELLM_MODEL_STRING else '*** NOT SET ***'}")

# --- Determine Target Path and Check Prerequisites ---
print("\n--- Test Setup: Determining Target Path & Checking Prerequisites ---")
SKIP_TESTS = False
SKIP_REASON = ""
TARGET_DETAILS = "Unknown"
AUTH_METHOD = "Unknown"

if not LITELLM_MODEL_STRING:
    SKIP_TESTS = True
    SKIP_REASON = "LITELLM_MODEL environment variable is required but not set."
    print(f"*** Test Skipping Reason: {SKIP_REASON} ***")
else:
    # Set the variable for the handler to use
    os.environ["LITELLM_MODEL"] = LITELLM_MODEL_STRING

    # --- Try importing the handler AFTER setting LITELLM_MODEL ---
    print("\n--- Test Setup: Attempting to Import llm_handler ---")
    llm_handler = None
    handler_import_error = None
    try:
        import llm_handler
        print("Successfully imported llm_handler.py")
    except Exception as e: # Catch broad exceptions during import
        handler_import_error = f"ERROR: Failed during llm_handler import: {e}"
        print(handler_import_error)
        # We might need dependencies even if handler imported partially
        llm_handler = None # Assume failure if any import exception happens

    if llm_handler is None:
        SKIP_TESTS = True
        SKIP_REASON = f"llm_handler.py could not be imported or initialized correctly. Import Error: {handler_import_error}"
        print(f"*** Test Skipping Reason: {SKIP_REASON} ***")
    else:
        # --- Path-Specific Checks ---
        is_azure_path = LITELLM_MODEL_STRING.lower().startswith("azure/")
        is_gemini_path = LITELLM_MODEL_STRING.lower().startswith("gemini/")
        is_openai_path = LITELLM_MODEL_STRING.lower().startswith("openai/")
        # Add more elif checks for other providers (Cohere, Anthropic, etc.) if needed

        if is_azure_path:
            print(f"Detected Azure Path (LITELLM_MODEL='{LITELLM_MODEL_STRING}')")
            AUTH_METHOD = "Azure AD"
            # Check Azure prerequisites
            ENDPOINT_URL = os.getenv("ENDPOINT_URL")
            API_VERSION = os.getenv("API_VERSION")
            DEPLOYMENT_NAME = LITELLM_MODEL_STRING.split('/', 1)[-1] # Get from model string

            if not getattr(llm_handler, 'AzureOpenAI', None) or \
               not getattr(llm_handler, 'DefaultAzureCredential', None):
                SKIP_TESTS = True
                SKIP_REASON = "Azure SDK Libraries (openai>=1.0.0, azure-identity) failed to import within llm_handler. Install them."
            elif not ENDPOINT_URL:
                SKIP_TESTS = True
                SKIP_REASON = "ENDPOINT_URL environment variable not set for Azure AD test path."
            elif not API_VERSION:
                SKIP_TESTS = True
                SKIP_REASON = "API_VERSION environment variable not set for Azure AD test path."
            else:
                # Ensure vars are set for the handler if read after import
                os.environ["ENDPOINT_URL"] = ENDPOINT_URL
                os.environ["API_VERSION"] = API_VERSION
                TARGET_DETAILS = f"Azure Deployment='{DEPLOYMENT_NAME}', Endpoint='{ENDPOINT_URL}', API='{API_VERSION}', Auth='{AUTH_METHOD}'"
                print("Azure AD path prerequisites met.")
                print("NOTE: Requires valid Azure credentials accessible by DefaultAzureCredential.")

        elif is_gemini_path:
            print(f"Detected Gemini/LiteLLM Path (LITELLM_MODEL='{LITELLM_MODEL_STRING}')")
            AUTH_METHOD = "API Key (GEMINI_API_KEY)"
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not getattr(llm_handler, 'litellm', None):
                 SKIP_TESTS = True
                 SKIP_REASON = "LiteLLM library failed to import within llm_handler. Install it."
            elif not GEMINI_API_KEY:
                SKIP_TESTS = True
                SKIP_REASON = "GEMINI_API_KEY environment variable not set for Gemini test path."
            else:
                TARGET_DETAILS = f"LiteLLM Model='{LITELLM_MODEL_STRING}', Auth='{AUTH_METHOD}'"
                print("Gemini/LiteLLM path prerequisites met.")

        elif is_openai_path:
            print(f"Detected OpenAI/LiteLLM Path (LITELLM_MODEL='{LITELLM_MODEL_STRING}')")
            AUTH_METHOD = "API Key (OPENAI_API_KEY)"
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            if not getattr(llm_handler, 'litellm', None):
                 SKIP_TESTS = True
                 SKIP_REASON = "LiteLLM library failed to import within llm_handler. Install it."
            elif not OPENAI_API_KEY:
                SKIP_TESTS = True
                SKIP_REASON = "OPENAI_API_KEY environment variable not set for OpenAI test path."
            else:
                TARGET_DETAILS = f"LiteLLM Model='{LITELLM_MODEL_STRING}', Auth='{AUTH_METHOD}'"
                print("OpenAI/LiteLLM path prerequisites met.")

        # Add elif blocks here for other providers like 'cohere/', 'anthropic/' etc.
        # checking for COHERE_API_KEY, ANTHROPIC_API_KEY respectively

        else:
            # Assume generic LiteLLM path, but we don't know which key to check explicitly
            # LiteLLM will fail internally if the required key isn't set.
            AUTH_METHOD = "API Key (Provider Specific)"
            print(f"Detected Generic LiteLLM Path (LITELLM_MODEL='{LITELLM_MODEL_STRING}')")
            if not getattr(llm_handler, 'litellm', None):
                 SKIP_TESTS = True
                 SKIP_REASON = "LiteLLM library failed to import within llm_handler. Install it."
            else:
                 TARGET_DETAILS = f"LiteLLM Model='{LITELLM_MODEL_STRING}', Auth='{AUTH_METHOD}'"
                 print("Generic LiteLLM path selected.")
                 print("NOTE: Requires the correct API key environment variable for this model to be set.")

        if SKIP_TESTS:
             print(f"*** Test Skipping Reason: {SKIP_REASON} ***")
        else:
             print("\n--- Test Setup: Pre-requisites met for the selected execution path ---")


# --- Mocking Setup (Identical to before) ---
print("\n--- Test Setup: Preparing Mocks ---")
class MockSample:
    def __init__(self, name="sample", graph=None, attributes=None): self.name=name; self.sample_graph=graph or {'B':['A'],'C':['B']}; self.attributes=attributes or {"attr_a":1,"attr_b":"hello"}
    def __dict__(self): return {"sample_graph":self.sample_graph,"attributes":self.attributes}
    def __str__(self): return f"MockSample(name='{self.name}',graph={self.sample_graph},attributes={self.attributes})"
def mock_build_sample_attribute_text(sample): print(f"      [Mock] Called mock_build_sample_attribute_text for sample: {sample.name}"); return f"Attribute A:{sample.attributes.get('attr_a','N/A')}\nAttribute B:{sample.attributes.get('attr_b','N/A')}"
def mock_get_graph_data(sample): print(f"      [Mock] Called mock_get_graph_data for sample: {sample.name}"); return sample.sample_graph
mock_dataset_module=MagicMock(); mock_dataset_module.build_sample_attribute_text=mock_build_sample_attribute_text; mock_dataset_module.get_graph_data=mock_get_graph_data
print("Mock dataset module created.")

# --- Helper function for pretty printing (Identical) ---
def print_test_summary(test_name, llm_details, prompt, raw_response, final_result, success_reason): print("\n"+"="*30+f" {test_name} Summary "+"="*30); print(f"[LLM Details]   : {llm_details}"); print(f"[Input Prompt]  :\n{textwrap.indent(json.dumps(prompt,indent=2),'  ')}"); print(f"[Raw Response]  :\n{textwrap.indent(str(raw_response),'  ')}"); print(f"[Final Result]  : {final_result}"); print(f"[Success Reason]: {success_reason}"); print("="*(62+len(test_name))); sys.stdout.flush()

# --- Test Class ---
print(f"\n--- Defining Test Class: TestLLMHandler (Skipped={SKIP_TESTS}, Reason='{SKIP_REASON}') ---")
@unittest.skipIf(SKIP_TESTS, SKIP_REASON)
class TestLLMHandler(unittest.TestCase): # Generic Class Name

    @classmethod
    def setUpClass(cls):
        """Set up shared resources for all tests in this class."""
        print("\n=========================================================")
        print(f"=== Running setUpClass for TestLLMHandler ===")
        print(f"=== Testing Path: {AUTH_METHOD} ===")
        print("=========================================================")
        cls.dataset_name = "mock_dynamic_dataset"
        # TARGET_DETAILS is set during the initial prerequisite check
        cls.llm_target_details = TARGET_DETAILS
        cls.good_sample = MockSample(name="good_sample", graph={'start':[],'process':['start'],'end':['process']}, attributes={'id':123,'status':'completed','value':42.5})
        cls.bad_graph_sample = MockSample(name="bad_graph_sample", graph={'A':['B'],'B':['A']}, attributes={'id':456,'status':'pending'})
        print(f"Created dataset_name: {cls.dataset_name}")
        print(f"Created good_sample: {cls.good_sample}")
        print(f"Created bad_graph_sample: {cls.bad_graph_sample}")
        print(f"LLM Target Details: {cls.llm_target_details}")
        print("setUpClass complete.")
        print("---------------------------------------------------------")
        sys.stdout.flush()

    @classmethod
    def tearDownClass(cls):
        print("\n=========================================================")
        print(f"=== Running tearDownClass for TestLLMHandler ===")
        print("=========================================================")
        sys.stdout.flush()

    # --- Test Methods ---
    # The actual test logic remains identical, as it calls the handler's
    # consistent interface. The setup ensures the correct backend is called.
    # The print_test_summary uses cls.llm_target_details set dynamically.

    @patch('importlib.import_module', return_value=mock_dataset_module)
    def test_01_llm_judge_sample(self, mock_import):
        """Test llm_judge_sample returns float [0..1] (Dynamic Path)."""
        test_name = "test_01_llm_judge_sample"
        print(f"\n--- Test Case: {test_name} ---")
        print(f"Testing function: llm_handler.llm_judge_sample (via {AUTH_METHOD} Path)")
        # ... (Rest of test_01 logic is identical to previous version) ...
        print(f"Using dataset_name: {self.dataset_name}")
        print(f"Using sample: {self.good_sample}")
        sys.stdout.flush()
        final_score = 0.0; prompt_msgs = []; raw_resp = "Error: Test setup failed"
        try:
            prompt_msgs = llm_handler.llm_judge_sample_prompt(self.dataset_name, self.good_sample)
            print(f"Generated Input Prompt: {json.dumps(prompt_msgs)}")
            print("Calling llm_handler.llm_judge_sample_with_debug (to get raw response)...")
            sys.stdout.flush()
            final_score, _ , raw_resp = llm_handler.llm_judge_sample_with_debug(self.dataset_name, self.good_sample)
            print(f"Received score: {final_score}")
            print(f"Received raw response snippet: '{raw_resp[:100]}...'")
            print("Asserting score is float between 0.0 and 1.0...")
            self.assertIsInstance(final_score, float); self.assertGreaterEqual(final_score, 0.0); self.assertLessEqual(final_score, 1.0)
            assertion_details = f"Score {final_score} is a float and within the range [0.0, 1.0]."
            print("Assertions passed.")
            print(f"Verifying mock import was called correctly..."); mock_import.assert_called_with(f"structpe.dataset.{self.dataset_name}_dataset"); print("Mock import verified.")
            print_test_summary(test_name=test_name, llm_details=self.llm_target_details, prompt=prompt_msgs, raw_response=raw_resp, final_result=f"Score = {final_score}", success_reason=assertion_details)
        except Exception as e:
            logging.error(f"Exception during {test_name}: {e}", exc_info=True)
            self.fail(f"Test failed due to unexpected exception: {e}")
        print(f"--- Test Case: {test_name} COMPLETE ---"); sys.stdout.flush()

    @patch('importlib.import_module', return_value=mock_dataset_module)
    def test_02_llm_judge_sample_with_debug(self, mock_import):
        """Test llm_judge_sample_with_debug (Dynamic Path)."""
        test_name = "test_02_llm_judge_sample_with_debug"
        print(f"\n--- Test Case: {test_name} ---")
        print(f"Testing function: llm_handler.llm_judge_sample_with_debug (via {AUTH_METHOD} Path)")
        # ... (Rest of test_02 logic is identical to previous version) ...
        print(f"Using dataset_name: {self.dataset_name}"); print(f"Using sample: {self.good_sample}"); print("Calling llm_handler.llm_judge_sample_with_debug..."); sys.stdout.flush()
        score, msgs, raw = (0.0, [], "Error: Test setup failed")
        try:
            score, msgs, raw = llm_handler.llm_judge_sample_with_debug(self.dataset_name, self.good_sample)
            print(f"Received result tuple: (Score={score}, Messages=..., Raw=...)"); print("Asserting result structure and types...")
            self.assertIsInstance((score, msgs, raw), tuple); self.assertEqual(len((score, msgs, raw)), 3)
            print(f"Asserting score ({score}) is float between 0.0 and 1.0..."); self.assertIsInstance(score, float); self.assertGreaterEqual(score, 0.0); self.assertLessEqual(score, 1.0)
            score_assertion = f"Score {score} is float and in [0.0, 1.0]."
            print(f"Asserting messages is a list..."); self.assertIsInstance(msgs, list); msgs_assertion = f"Messages is a list (length {len(msgs)})."
            print(f"Asserting raw response is a string..."); self.assertIsInstance(raw, str); raw_assertion = f"Raw response is a string (length {len(raw)})."
            if not raw: logging.warning("Raw response string is empty.")
            print("Assertions passed.")
            print(f"Verifying mock import was called correctly..."); mock_import.assert_called_with(f"structpe.dataset.{self.dataset_name}_dataset"); print("Mock import verified.")
            print_test_summary(test_name=test_name, llm_details=self.llm_target_details, prompt=msgs, raw_response=raw, final_result=f"Score = {score}", success_reason=f"{score_assertion} {msgs_assertion} {raw_assertion}")
        except Exception as e:
            logging.error(f"Exception during {test_name}: {e}", exc_info=True)
            self.fail(f"Test failed due to unexpected exception: {e}")
        print(f"--- Test Case: {test_name} COMPLETE ---"); sys.stdout.flush()

    @patch('importlib.import_module', return_value=mock_dataset_module)
    def test_03_llm_judge_graph(self, mock_import):
        """Test llm_judge_graph returns float [1..5] (Dynamic Path)."""
        test_name = "test_03_llm_judge_graph"
        print(f"\n--- Test Case: {test_name} ---")
        print(f"Testing function: llm_handler.llm_judge_graph (via {AUTH_METHOD} Path)")
        # ... (Rest of test_03 logic is identical to previous version, including the corrected call count check) ...
        score_good, score_bad = 3.0, 3.0; prompt_good, prompt_bad = [], []; raw_good, raw_bad = "Error: Test setup failed", "Error: Test setup failed"
        try:
            print(f"\nTesting with good graph sample: {self.good_sample}")
            prompt_good = llm_handler.llm_judge_graph_prompt(self.dataset_name, self.good_sample)
            print(f"Generated Input Prompt (Good Graph): {json.dumps(prompt_good)}")
            print("Calling llm_handler.llm_judge_graph_with_debug (good graph)..."); sys.stdout.flush()
            score_good, _, raw_good = llm_handler.llm_judge_graph_with_debug(self.dataset_name, self.good_sample)
            print(f"Received score (good graph): {score_good}"); print(f"Received raw response snippet (good graph): '{raw_good[:100]}...'")
            print("Asserting score is float between 1.0 and 5.0..."); self.assertIsInstance(score_good, float); self.assertGreaterEqual(score_good, 1.0); self.assertLessEqual(score_good, 5.0)
            good_assertion = f"Good graph score {score_good} is float and in [1.0, 5.0]."; print("Assertions passed for good graph.")
            print_test_summary(test_name=f"{test_name} (Good Graph)", llm_details=self.llm_target_details, prompt=prompt_good, raw_response=raw_good, final_result=f"Score = {score_good}", success_reason=good_assertion)

            print(f"\nTesting with bad graph sample: {self.bad_graph_sample}")
            prompt_bad = llm_handler.llm_judge_graph_prompt(self.dataset_name, self.bad_graph_sample)
            print(f"Generated Input Prompt (Bad Graph): {json.dumps(prompt_bad)}")
            print("Calling llm_handler.llm_judge_graph_with_debug (bad graph)..."); sys.stdout.flush()
            score_bad, _, raw_bad = llm_handler.llm_judge_graph_with_debug(self.dataset_name, self.bad_graph_sample)
            print(f"Received score (bad graph): {score_bad}"); print(f"Received raw response snippet (bad graph): '{raw_bad[:100]}...'")
            print("Asserting score is float between 1.0 and 5.0..."); self.assertIsInstance(score_bad, float); self.assertGreaterEqual(score_bad, 1.0); self.assertLessEqual(score_bad, 5.0)
            bad_assertion = f"Bad graph score {score_bad} is float and in [1.0, 5.0]."; print("Assertions passed for bad graph.")
            print("(Note: LLM judgment on graph correctness can be subjective)")
            print_test_summary(test_name=f"{test_name} (Bad Graph)", llm_details=self.llm_target_details, prompt=prompt_bad, raw_response=raw_bad, final_result=f"Score = {score_bad}", success_reason=bad_assertion)

            print(f"\nVerifying mock import was called 4 times..."); self.assertEqual(mock_import.call_count, 4, f"Expected mock_import to be called 4 times, but was called {mock_import.call_count} times.")
            mock_import.assert_any_call(f"structpe.dataset.{self.dataset_name}_dataset"); print("Mock import verified.")
        except Exception as e:
            logging.error(f"Exception during {test_name}: {e}", exc_info=True)
            self.fail(f"Test failed due to unexpected exception: {e}")
        print(f"--- Test Case: {test_name} COMPLETE ---"); sys.stdout.flush()

    @patch('importlib.import_module', return_value=mock_dataset_module)
    def test_04_llm_judge_graph_with_debug(self, mock_import):
        """Test llm_judge_graph_with_debug (Dynamic Path)."""
        test_name = "test_04_llm_judge_graph_with_debug"
        print(f"\n--- Test Case: {test_name} ---")
        print(f"Testing function: llm_handler.llm_judge_graph_with_debug (via {AUTH_METHOD} Path)")
        # ... (Rest of test_04 logic is identical to previous version) ...
        print(f"Using dataset_name: {self.dataset_name}"); print(f"Using sample: {self.good_sample}"); print("Calling llm_handler.llm_judge_graph_with_debug..."); sys.stdout.flush()
        score, msgs, raw = (3.0, [], "Error: Test setup failed")
        try:
            score, msgs, raw = llm_handler.llm_judge_graph_with_debug(self.dataset_name, self.good_sample)
            print(f"Received result tuple: (Score={score}, Messages=..., Raw=...)"); print("Asserting result structure and types...")
            self.assertIsInstance((score, msgs, raw), tuple); self.assertEqual(len((score, msgs, raw)), 3)
            print(f"Asserting score ({score}) is float between 1.0 and 5.0..."); self.assertIsInstance(score, float); self.assertGreaterEqual(score, 1.0); self.assertLessEqual(score, 5.0)
            score_assertion = f"Score {score} is float and in [1.0, 5.0]."
            print(f"Asserting messages is a list..."); self.assertIsInstance(msgs, list); msgs_assertion = f"Messages is a list (length {len(msgs)})."
            print(f"Asserting raw response is a string..."); self.assertIsInstance(raw, str); raw_assertion = f"Raw response is a string (length {len(raw)})."
            if not raw: logging.warning("Raw response string is empty.")
            print("Assertions passed.")
            print(f"Verifying mock import was called correctly..."); mock_import.assert_called_with(f"structpe.dataset.{self.dataset_name}_dataset"); print("Mock import verified.")
            print_test_summary(test_name=test_name, llm_details=self.llm_target_details, prompt=msgs, raw_response=raw, final_result=f"Score = {score}", success_reason=f"{score_assertion} {msgs_assertion} {raw_assertion}")
        except Exception as e:
            logging.error(f"Exception during {test_name}: {e}", exc_info=True)
            self.fail(f"Test failed due to unexpected exception: {e}")
        print(f"--- Test Case: {test_name} COMPLETE ---"); sys.stdout.flush()

    def test_05_llm_check_graph_interface(self):
        """Test llm_check_graph returns bool, str (Dynamic Path)."""
        test_name = "test_05_llm_check_graph_interface"
        print(f"\n--- Test Case: {test_name} ---")
        print(f"Testing function: llm_handler.llm_check_graph (via {AUTH_METHOD} Path)")
        # ... (Rest of test_05 logic is identical to previous version) ...
        graph_str_good = json.dumps(self.good_sample.sample_graph); prompt_good_str = f"\nWe have a sample with a 'sample_graph' adjacency information: {graph_str_good}\nCheck if it potentially forms a valid directed acyclic graph (DAG) with no cycles\nand makes logical sense for attribute generation order based on typical data dependencies.\nReply ONLY with the word \"OK\" if it seems valid, or \"FAIL\" if it seems invalid or cyclic.\nFollow the word \"OK\" or \"FAIL\" with a very brief explanation on a new line.\n"; prompt_good = [{"role":"user","content":prompt_good_str.strip()}]
        graph_str_bad = json.dumps(self.bad_graph_sample.sample_graph); prompt_bad_str = f"\nWe have a sample with a 'sample_graph' adjacency information: {graph_str_bad}\nCheck if it potentially forms a valid directed acyclic graph (DAG) with no cycles\nand makes logical sense for attribute generation order based on typical data dependencies.\nReply ONLY with the word \"OK\" if it seems valid, or \"FAIL\" if it seems invalid or cyclic.\nFollow the word \"OK\" or \"FAIL\" with a very brief explanation on a new line.\n"; prompt_bad = [{"role":"user","content":prompt_bad_str.strip()}]
        is_ok_good, explanation_good = False, "Error: Test setup failed"; is_ok_bad, explanation_bad = True, "Error: Test setup failed"
        try:
            print(f"\nTesting with good graph sample: {self.good_sample.name}"); print(f"Effective Input Prompt (Good Graph): {json.dumps(prompt_good)}"); print("Calling llm_handler.llm_check_graph..."); sys.stdout.flush()
            is_ok_good, explanation_good = llm_handler.llm_check_graph(self.good_sample); print(f"Received (good graph): OK={is_ok_good}, Explanation='{explanation_good}'"); print("Asserting return types (bool, str)..."); self.assertIsInstance(is_ok_good, bool); self.assertIsInstance(explanation_good, str)
            good_assertion = f"Result ({is_ok_good}, <explanation string>) matches expected types (bool, str)."; print("Assertions passed for good graph.")
            print_test_summary(test_name=f"{test_name} (Good Graph)", llm_details=self.llm_target_details, prompt=prompt_good, raw_response=explanation_good, final_result=f"Is OK = {is_ok_good}", success_reason=good_assertion)

            print(f"\nTesting with bad graph sample: {self.bad_graph_sample.name}"); print(f"Effective Input Prompt (Bad Graph): {json.dumps(prompt_bad)}"); print("Calling llm_handler.llm_check_graph..."); sys.stdout.flush()
            is_ok_bad, explanation_bad = llm_handler.llm_check_graph(self.bad_graph_sample); print(f"Received (bad graph): OK={is_ok_bad}, Explanation='{explanation_bad}'"); print("Asserting return types (bool, str)..."); self.assertIsInstance(is_ok_bad, bool); self.assertIsInstance(explanation_bad, str)
            bad_assertion = f"Result ({is_ok_bad}, <explanation string>) matches expected types (bool, str)."; print("Assertions passed for bad graph.")
            print("(Note: LLM agreement on OK/FAIL for graph checks depends heavily on the model and prompt adherence)")
            print_test_summary(test_name=f"{test_name} (Bad Graph)", llm_details=self.llm_target_details, prompt=prompt_bad, raw_response=explanation_bad, final_result=f"Is OK = {is_ok_bad}", success_reason=bad_assertion)
        except Exception as e:
            logging.error(f"Exception during {test_name}: {e}", exc_info=True)
            self.fail(f"Test failed due to unexpected exception: {e}")
        print(f"--- Test Case: {test_name} COMPLETE ---"); sys.stdout.flush()


if __name__ == '__main__':
    print("\n--- Running Unittests ---")
    # Usage: Set LITELLM_MODEL and relevant API keys/config, then run:
    # python -m unittest test_llm_handler.py
    unittest.main(verbosity=2)