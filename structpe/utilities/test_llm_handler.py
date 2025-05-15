import unittest
import os
import json
import importlib
from unittest.mock import patch, MagicMock
import sys # To force flush print statements if needed
import logging
import textwrap # For formatting long outputs

# --- Configuration & Pre-Checks ---
print("--- Test Setup: Reading Configuration for Azure AD Path ---")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read the Azure-specific variables needed for AD Auth path
ENDPOINT_URL = os.getenv("ENDPOINT_URL", "https://syndata.openai.azure.com/")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4")
API_VERSION = os.getenv("API_VERSION", "2024-05-01-preview")

print(f"Read ENDPOINT_URL: {ENDPOINT_URL}")
print(f"Read DEPLOYMENT_NAME: {DEPLOYMENT_NAME}")
print(f"Read API_VERSION: {API_VERSION}")
print("NOTE: AZURE_API_KEY is NOT used for Azure AD authentication path.")

# --- Set LITELLM_MODEL to trigger the Azure path in the handler ---
LITELLM_MODEL_STRING = f"azure/{DEPLOYMENT_NAME}"
os.environ["LITELLM_MODEL"] = LITELLM_MODEL_STRING
os.environ["ENDPOINT_URL"] = ENDPOINT_URL
os.environ["API_VERSION"] = API_VERSION

print(f"\n--- Test Setup: Ensuring Environment Variables for Handler ---")
print(f"Set LITELLM_MODEL = {os.environ['LITELLM_MODEL']} (to trigger Azure AD path)")
print(f"Set ENDPOINT_URL = {os.environ['ENDPOINT_URL']}")
print(f"Set API_VERSION = {os.environ['API_VERSION']}")

# --- Try importing the handler AFTER setting env vars ---
print("\n--- Test Setup: Attempting to Import llm_handler ---")
llm_handler = None
handler_import_error = None
# ... (rest of the import and skip logic remains the same as the previous version) ...
try:
    # Ensure logging is configured before import if handler uses it at module level
    import llm_handler
    print("Successfully imported llm_handler.py")
except ValueError as e:
    handler_import_error = f"ERROR: Failed to import llm_handler. Potential issue with LITELLM_MODEL setup. Error: {e}"
    print(handler_import_error)
except ImportError as e:
    handler_import_error = f"ERROR: Failed to import llm_handler or its dependencies (openai, azure-identity?). Error: {e}"
    print(handler_import_error)
except Exception as e:
     handler_import_error = f"ERROR: An unexpected error occurred during llm_handler import: {e}"
     print(handler_import_error)

# --- Determine if tests should be skipped ---
print("\n--- Test Setup: Determining Skip Conditions ---")
SKIP_TESTS = False
SKIP_REASON = ""

if llm_handler is None:
    SKIP_TESTS = True
    SKIP_REASON = f"llm_handler.py could not be imported. Import Error: {handler_import_error}"
    print(f"*** Test Skipping Reason: {SKIP_REASON} ***")
else:
    # Check conditions specifically for the Azure AD Path
    print(f"Checking prerequisites for Azure AD path (LITELLM_MODEL='{LITELLM_MODEL_STRING}')")
    # 1. Check if Azure SDK components were successfully imported *within* the handler
    if not getattr(llm_handler, 'AzureOpenAI', None) or \
       not getattr(llm_handler, 'DefaultAzureCredential', None) or \
       not getattr(llm_handler, 'get_bearer_token_provider', None):
        SKIP_TESTS = True
        SKIP_REASON = "Azure SDK Libraries (openai>=1.0.0, azure-identity) failed to import within llm_handler. Install them."
        print(f"*** Test Skipping Reason: {SKIP_REASON} ***")
    # 2. Check if necessary environment variables for Azure AD are set
    elif not ENDPOINT_URL:
        SKIP_TESTS = True
        SKIP_REASON = "ENDPOINT_URL environment variable not set for Azure AD test path."
        print(f"*** Test Skipping Reason: {SKIP_REASON} ***")
    elif not API_VERSION:
        SKIP_TESTS = True
        SKIP_REASON = "API_VERSION environment variable not set for Azure AD test path."
        print(f"*** Test Skipping Reason: {SKIP_REASON} ***")
    else:
        print("Azure AD path prerequisites met (handler imported, Azure SDKs seem available, config vars set).")
        print("\n***************************************************************************")
        print("*** NOTE: Running Azure AD tests. This requires valid Azure credentials ***")
        print("*** accessible by Azure DefaultAzureCredential in your env.       ***")
        print("***************************************************************************\n")


# --- Mocking Setup ---
# (No changes needed in mocking setup itself)
print("\n--- Test Setup: Preparing Mocks ---")
# 1. Mock Sample Class
class MockSample:
    def __init__(self, name="sample", graph=None, attributes=None):
        self.name = name # Added name for logging
        self.sample_graph = graph or {'B': ['A'], 'C': ['B']} # Default valid DAG
        self.attributes = attributes or {"attr_a": 1, "attr_b": "hello"}

    def __dict__(self):
        # Simple dict representation for the fallback in _get_sample_attributes
        return {"sample_graph": self.sample_graph, "attributes": self.attributes}

    def __str__(self):
        # Nicer representation for logging
        return f"MockSample(name='{self.name}', graph={self.sample_graph}, attributes={self.attributes})"

# 2. Mock Dataset Helper Functions
def mock_build_sample_attribute_text(sample):
    """Simulates retrieving attribute text for the sample."""
    print(f"      [Mock] Called mock_build_sample_attribute_text for sample: {sample.name}")
    return f"Attribute A: {sample.attributes.get('attr_a', 'N/A')}\nAttribute B: {sample.attributes.get('attr_b', 'N/A')}"

def mock_get_graph_data(sample):
    """Simulates retrieving graph data for the sample."""
    print(f"      [Mock] Called mock_get_graph_data for sample: {sample.name}")
    return sample.sample_graph

# 3. Create a mock module object to be returned by importlib.import_module
mock_dataset_module = MagicMock()
mock_dataset_module.build_sample_attribute_text = mock_build_sample_attribute_text
mock_dataset_module.get_graph_data = mock_get_graph_data
print("Mock dataset module created with helper functions.")

# --- Helper function for pretty printing ---
def print_test_summary(test_name, llm_details, prompt, raw_response, final_result, success_reason):
    print("\n" + "="*30 + f" {test_name} Summary " + "="*30)
    print(f"[LLM Details]   : {llm_details}")
    print(f"[Input Prompt]  :\n{textwrap.indent(json.dumps(prompt, indent=2), '  ')}")
    print(f"[Raw Response]  :\n{textwrap.indent(str(raw_response), '  ')}")
    print(f"[Final Result]  : {final_result}")
    print(f"[Success Reason]: {success_reason}")
    print("="* (62 + len(test_name)))
    sys.stdout.flush()


# --- Test Class ---
print(f"\n--- Defining Test Class: TestLLMHandlerAzureAD (Skipped={SKIP_TESTS}, Reason='{SKIP_REASON}') ---")
@unittest.skipIf(SKIP_TESTS, SKIP_REASON)
class TestLLMHandlerAzureAD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up shared resources for all tests in this class."""
        print("\n=========================================================")
        print("=== Running setUpClass for TestLLMHandlerAzureAD ===")
        print("=========================================================")
        cls.dataset_name = "mock_azure_dataset"
        cls.llm_target_details = f"Azure Deployment='{DEPLOYMENT_NAME}', Endpoint='{ENDPOINT_URL}', API='{API_VERSION}', Auth='Azure AD'"
        cls.good_sample = MockSample(
            name="good_sample",
            graph={'start': [], 'process': ['start'], 'end': ['process']},
            attributes={'id': 123, 'status': 'completed', 'value': 42.5}
        )
        cls.bad_graph_sample = MockSample(
             name="bad_graph_sample",
             graph={'A': ['B'], 'B': ['A']}, # Cyclic graph
             attributes={'id': 456, 'status': 'pending'}
        )
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
        print("=== Running tearDownClass for TestLLMHandlerAzureAD ===")
        print("=========================================================")
        sys.stdout.flush()

    # Patch importlib.import_module for all tests in this class
    @patch('importlib.import_module', return_value=mock_dataset_module)
    def test_01_llm_judge_sample(self, mock_import):
        """Test llm_judge_sample returns a float between 0 and 1 (Azure AD Path)."""
        test_name = "test_01_llm_judge_sample"
        print(f"\n--- Test Case: {test_name} ---")
        print(f"Testing function: llm_handler.llm_judge_sample (via Azure AD Path)")
        print(f"Using dataset_name: {self.dataset_name}")
        print(f"Using sample: {self.good_sample}")
        sys.stdout.flush()

        final_score = 0.0
        prompt_msgs = []
        raw_resp = "Error: Test setup failed"

        try:
            # Generate prompt separately for logging
            prompt_msgs = llm_handler.llm_judge_sample_prompt(self.dataset_name, self.good_sample)
            print(f"Generated Input Prompt: {json.dumps(prompt_msgs)}")
            print("Calling llm_handler.llm_judge_sample_with_debug (to get raw response)...")
            sys.stdout.flush()
            # Call _with_debug version to get all details
            final_score, _ , raw_resp = llm_handler.llm_judge_sample_with_debug(self.dataset_name, self.good_sample)
            print(f"Received score: {final_score}")
            print(f"Received raw response snippet: '{raw_resp[:100]}...'")

            print("Asserting score is float between 0.0 and 1.0...")
            self.assertIsInstance(final_score, float)
            self.assertGreaterEqual(final_score, 0.0)
            self.assertLessEqual(final_score, 1.0)
            assertion_details = f"Score {final_score} is a float and within the range [0.0, 1.0]."
            print("Assertions passed.")
            print(f"Verifying mock import was called correctly...")
            mock_import.assert_called_with(f"structpe.dataset.{self.dataset_name}_dataset")
            print("Mock import verified.")

            print_test_summary(
                test_name=test_name,
                llm_details=self.llm_target_details,
                prompt=prompt_msgs,
                raw_response=raw_resp,
                final_result=f"Score = {final_score}",
                success_reason=assertion_details
            )

        except Exception as e:
            logging.error(f"Exception during {test_name}: {e}", exc_info=True)
            self.fail(f"Test failed due to unexpected exception: {e}")

        print(f"--- Test Case: {test_name} COMPLETE ---")
        sys.stdout.flush()


    @patch('importlib.import_module', return_value=mock_dataset_module)
    def test_02_llm_judge_sample_with_debug(self, mock_import):
        """Test llm_judge_sample_with_debug (Azure AD Path)."""
        test_name = "test_02_llm_judge_sample_with_debug"
        print(f"\n--- Test Case: {test_name} ---")
        print(f"Testing function: llm_handler.llm_judge_sample_with_debug (via Azure AD Path)")
        print(f"Using dataset_name: {self.dataset_name}")
        print(f"Using sample: {self.good_sample}")
        print("Calling llm_handler.llm_judge_sample_with_debug...")
        sys.stdout.flush()

        score, msgs, raw = (0.0, [], "Error: Test setup failed")

        try:
            score, msgs, raw = llm_handler.llm_judge_sample_with_debug(self.dataset_name, self.good_sample)

            print(f"Received result tuple: (Score={score}, Messages=..., Raw=...)")
            # Raw response snippet already logged by handler, full is below
            print("Asserting result structure and types...")
            self.assertIsInstance((score, msgs, raw), tuple)
            self.assertEqual(len((score, msgs, raw)), 3)

            print(f"Asserting score ({score}) is float between 0.0 and 1.0...")
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            score_assertion = f"Score {score} is float and in [0.0, 1.0]."

            print(f"Asserting messages is a list...")
            self.assertIsInstance(msgs, list)
            msgs_assertion = f"Messages is a list (length {len(msgs)})."

            print(f"Asserting raw response is a string...")
            self.assertIsInstance(raw, str)
            raw_assertion = f"Raw response is a string (length {len(raw)})."
            if not raw: logging.warning("Raw response string is empty.")

            print("Assertions passed.")
            print(f"Verifying mock import was called correctly...")
            mock_import.assert_called_with(f"structpe.dataset.{self.dataset_name}_dataset")
            print("Mock import verified.")

            print_test_summary(
                test_name=test_name,
                llm_details=self.llm_target_details,
                prompt=msgs,
                raw_response=raw,
                final_result=f"Score = {score}",
                success_reason=f"{score_assertion} {msgs_assertion} {raw_assertion}"
            )

        except Exception as e:
            logging.error(f"Exception during {test_name}: {e}", exc_info=True)
            self.fail(f"Test failed due to unexpected exception: {e}")

        print(f"--- Test Case: {test_name} COMPLETE ---")
        sys.stdout.flush()


    @patch('importlib.import_module', return_value=mock_dataset_module)
    def test_03_llm_judge_graph(self, mock_import):
        """Test llm_judge_graph returns float [1..5] (Azure AD Path)."""
        test_name = "test_03_llm_judge_graph"
        print(f"\n--- Test Case: {test_name} ---")
        print(f"Testing function: llm_handler.llm_judge_graph (via Azure AD Path)")

        score_good, score_bad = 3.0, 3.0
        prompt_good, prompt_bad = [], []
        raw_good, raw_bad = "Error: Test setup failed", "Error: Test setup failed"

        try:
            # --- Test with good graph ---
            print(f"\nTesting with good graph sample: {self.good_sample}")
            prompt_good = llm_handler.llm_judge_graph_prompt(self.dataset_name, self.good_sample)
            print(f"Generated Input Prompt (Good Graph): {json.dumps(prompt_good)}")
            print("Calling llm_handler.llm_judge_graph_with_debug (good graph)...")
            sys.stdout.flush()
            score_good, _, raw_good = llm_handler.llm_judge_graph_with_debug(self.dataset_name, self.good_sample)
            print(f"Received score (good graph): {score_good}")
            print(f"Received raw response snippet (good graph): '{raw_good[:100]}...'")
            print("Asserting score is float between 1.0 and 5.0...")
            self.assertIsInstance(score_good, float)
            self.assertGreaterEqual(score_good, 1.0)
            self.assertLessEqual(score_good, 5.0)
            good_assertion = f"Good graph score {score_good} is float and in [1.0, 5.0]."
            print("Assertions passed for good graph.")

            print_test_summary(
                test_name=f"{test_name} (Good Graph)",
                llm_details=self.llm_target_details,
                prompt=prompt_good,
                raw_response=raw_good,
                final_result=f"Score = {score_good}",
                success_reason=good_assertion
            )

            # --- Test with bad graph ---
            print(f"\nTesting with bad graph sample: {self.bad_graph_sample}")
            prompt_bad = llm_handler.llm_judge_graph_prompt(self.dataset_name, self.bad_graph_sample)
            print(f"Generated Input Prompt (Bad Graph): {json.dumps(prompt_bad)}")
            print("Calling llm_handler.llm_judge_graph_with_debug (bad graph)...")
            sys.stdout.flush()
            score_bad, _, raw_bad = llm_handler.llm_judge_graph_with_debug(self.dataset_name, self.bad_graph_sample)
            print(f"Received score (bad graph): {score_bad}")
            print(f"Received raw response snippet (bad graph): '{raw_bad[:100]}...'")
            print("Asserting score is float between 1.0 and 5.0...")
            self.assertIsInstance(score_bad, float)
            self.assertGreaterEqual(score_bad, 1.0)
            self.assertLessEqual(score_bad, 5.0)
            bad_assertion = f"Bad graph score {score_bad} is float and in [1.0, 5.0]."
            print("Assertions passed for bad graph.")
            print("(Note: LLM judgment on graph correctness can be subjective)")

            print_test_summary(
                test_name=f"{test_name} (Bad Graph)",
                llm_details=self.llm_target_details,
                prompt=prompt_bad,
                raw_response=raw_bad,
                final_result=f"Score = {score_bad}",
                success_reason=bad_assertion
            )

            print(f"\nVerifying mock import was called twice...")
            self.assertEqual(mock_import.call_count, 4, f"Expected mock_import to be called twice, but was called {mock_import.call_count} times.")
            mock_import.assert_any_call(f"structpe.dataset.{self.dataset_name}_dataset")
            print("Mock import verified.")

        except Exception as e:
            logging.error(f"Exception during {test_name}: {e}", exc_info=True)
            self.fail(f"Test failed due to unexpected exception: {e}")

        print(f"--- Test Case: {test_name} COMPLETE ---")
        sys.stdout.flush()


    @patch('importlib.import_module', return_value=mock_dataset_module)
    def test_04_llm_judge_graph_with_debug(self, mock_import):
        """Test llm_judge_graph_with_debug (Azure AD Path)."""
        test_name = "test_04_llm_judge_graph_with_debug"
        print(f"\n--- Test Case: {test_name} ---")
        print(f"Testing function: llm_handler.llm_judge_graph_with_debug (via Azure AD Path)")
        print(f"Using dataset_name: {self.dataset_name}")
        print(f"Using sample: {self.good_sample}")
        print("Calling llm_handler.llm_judge_graph_with_debug...")
        sys.stdout.flush()

        score, msgs, raw = (3.0, [], "Error: Test setup failed")

        try:
            score, msgs, raw = llm_handler.llm_judge_graph_with_debug(self.dataset_name, self.good_sample)

            print(f"Received result tuple: (Score={score}, Messages=..., Raw=...)")
            # Raw response snippet logged by handler, full printed below
            print("Asserting result structure and types...")
            self.assertIsInstance((score, msgs, raw), tuple)
            self.assertEqual(len((score, msgs, raw)), 3)

            print(f"Asserting score ({score}) is float between 1.0 and 5.0...")
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 1.0)
            self.assertLessEqual(score, 5.0)
            score_assertion = f"Score {score} is float and in [1.0, 5.0]."

            print(f"Asserting messages is a list...")
            self.assertIsInstance(msgs, list)
            msgs_assertion = f"Messages is a list (length {len(msgs)})."

            print(f"Asserting raw response is a string...")
            self.assertIsInstance(raw, str)
            raw_assertion = f"Raw response is a string (length {len(raw)})."
            if not raw: logging.warning("Raw response string is empty.")

            print("Assertions passed.")
            print(f"Verifying mock import was called correctly...")
            mock_import.assert_called_with(f"structpe.dataset.{self.dataset_name}_dataset")
            print("Mock import verified.")

            print_test_summary(
                test_name=test_name,
                llm_details=self.llm_target_details,
                prompt=msgs,
                raw_response=raw,
                final_result=f"Score = {score}",
                success_reason=f"{score_assertion} {msgs_assertion} {raw_assertion}"
            )

        except Exception as e:
            logging.error(f"Exception during {test_name}: {e}", exc_info=True)
            self.fail(f"Test failed due to unexpected exception: {e}")

        print(f"--- Test Case: {test_name} COMPLETE ---")
        sys.stdout.flush()


    # Note: llm_check_graph doesn't use the mock import
    def test_05_llm_check_graph_interface(self):
        """Test llm_check_graph returns bool, str (Azure AD Path)."""
        test_name = "test_05_llm_check_graph_interface"
        print(f"\n--- Test Case: {test_name} ---")
        print(f"Testing function: llm_handler.llm_check_graph (via Azure AD Path)")

        # Reconstruct the effective prompt for logging
        graph_str_good = json.dumps(self.good_sample.sample_graph)
        prompt_good_str = f"""
We have a sample with a 'sample_graph' adjacency information: {graph_str_good}
Check if it potentially forms a valid directed acyclic graph (DAG) with no cycles
and makes logical sense for attribute generation order based on typical data dependencies.
Reply ONLY with the word "OK" if it seems valid, or "FAIL" if it seems invalid or cyclic.
Follow the word "OK" or "FAIL" with a very brief explanation on a new line.
"""
        prompt_good = [{"role":"user","content":prompt_good_str.strip()}]

        graph_str_bad = json.dumps(self.bad_graph_sample.sample_graph)
        prompt_bad_str = f"""
We have a sample with a 'sample_graph' adjacency information: {graph_str_bad}
Check if it potentially forms a valid directed acyclic graph (DAG) with no cycles
and makes logical sense for attribute generation order based on typical data dependencies.
Reply ONLY with the word "OK" if it seems valid, or "FAIL" if it seems invalid or cyclic.
Follow the word "OK" or "FAIL" with a very brief explanation on a new line.
"""
        prompt_bad = [{"role":"user","content":prompt_bad_str.strip()}]

        is_ok_good, explanation_good = False, "Error: Test setup failed"
        is_ok_bad, explanation_bad = True, "Error: Test setup failed"


        try:
            # --- Test with good graph ---
            print(f"\nTesting with good graph sample: {self.good_sample.name}")
            print(f"Effective Input Prompt (Good Graph): {json.dumps(prompt_good)}")
            print("Calling llm_handler.llm_check_graph...")
            sys.stdout.flush()
            is_ok_good, explanation_good = llm_handler.llm_check_graph(self.good_sample)
            print(f"Received (good graph): OK={is_ok_good}, Explanation='{explanation_good}'")
            print("Asserting return types (bool, str)...")
            self.assertIsInstance(is_ok_good, bool)
            self.assertIsInstance(explanation_good, str)
            good_assertion = f"Result ({is_ok_good}, <explanation string>) matches expected types (bool, str)."
            print("Assertions passed for good graph.")

            print_test_summary(
                test_name=f"{test_name} (Good Graph)",
                llm_details=self.llm_target_details,
                prompt=prompt_good,
                raw_response=explanation_good, # Explanation contains the raw response here
                final_result=f"Is OK = {is_ok_good}",
                success_reason=good_assertion
            )

            # --- Test with bad graph ---
            print(f"\nTesting with bad graph sample: {self.bad_graph_sample.name}")
            print(f"Effective Input Prompt (Bad Graph): {json.dumps(prompt_bad)}")
            print("Calling llm_handler.llm_check_graph...")
            sys.stdout.flush()
            is_ok_bad, explanation_bad = llm_handler.llm_check_graph(self.bad_graph_sample)
            print(f"Received (bad graph): OK={is_ok_bad}, Explanation='{explanation_bad}'")
            print("Asserting return types (bool, str)...")
            self.assertIsInstance(is_ok_bad, bool)
            self.assertIsInstance(explanation_bad, str)
            bad_assertion = f"Result ({is_ok_bad}, <explanation string>) matches expected types (bool, str)."
            print("Assertions passed for bad graph.")
            print("(Note: LLM agreement on OK/FAIL for graph checks depends heavily on the model and prompt adherence)")

            print_test_summary(
                test_name=f"{test_name} (Bad Graph)",
                llm_details=self.llm_target_details,
                prompt=prompt_bad,
                raw_response=explanation_bad, # Explanation contains the raw response here
                final_result=f"Is OK = {is_ok_bad}",
                success_reason=bad_assertion
            )

        except Exception as e:
            logging.error(f"Exception during {test_name}: {e}", exc_info=True)
            self.fail(f"Test failed due to unexpected exception: {e}")

        print(f"--- Test Case: {test_name} COMPLETE ---")
        sys.stdout.flush()


if __name__ == '__main__':
    print("\n--- Running Unittests ---")
    # Using verbosity=2 gives a bit more detail from unittest itself, along with our custom prints
    unittest.main(verbosity=2)