"""
test_pipeline.py: Tests the pipeline command by providing a sample JSON 
and checking the result. 
"""

import unittest
import os
import json
import subprocess

class TestPipeline(unittest.TestCase):
    def test_pipeline_with_json(self):
        """
        We'll run the 'structpe pipeline --json-file=tests/pipeline_data.json' 
        via subprocess to ensure it runs end-to-end with no errors.
        """
        pipeline_file = os.path.join(os.path.dirname(__file__), "pipeline_data.json")
        cmd = ["structpe", "pipeline", "--json-file", pipeline_file]

        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Pipeline failed with stderr: {result.stderr}")
        self.assertIn("=== Pipeline Evaluate Results ===", result.stdout)

    def test_pipeline_json_structure(self):
        """
        We'll also check that pipeline_data.json has the expected fields.
        """
        pipeline_file = os.path.join(os.path.dirname(__file__), "pipeline_data.json")
        with open(pipeline_file, "r") as f:
            data = json.load(f)
        self.assertIn("dataset_name", data)
        self.assertIn("samples", data)

if __name__ == "__main__":
    unittest.main()
