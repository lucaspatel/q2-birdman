import unittest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from birdman_cli import cli 

BIOM_PATH = "data/94270_filtered.biom"
METADATA_PATH = "path/to/metadata/file.tsv"
FORMULA = "Species ~ Condition"
OUTPUT_DIR = "path/to/output/dir"
INPUT_DIR = "path/to/input/dir"
OMIC = "microbiome"
THREADS = 4
INPUT_PATH = "path/to/input.tsv"
VARIABLES = "var1,var2"
FEATURE_METADATA = "path/to/feature/metadata.tsv"

class TestCLIRun(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.biom_path = "../data/94270_filtered.biom"
        self.metadata_path = "../data/11913_filtered.tsv"
        self.formula = "age"
        self.output_dir = "../test_out"
        self.email = "foo@bar.com"

    @patch('subprocess.run')
    def test_run_success(self, mock_run):
        """Test the run function handles successful submission."""
        mock_run.return_value = MagicMock(returncode=0, stdout='Job submitted successfully')

        result = self.runner.invoke(cli, ['run', '-i', self.biom_path, '-m', self.metadata_path, '-f', self.formula, '-o', self.output_dir, '-e', self.email])
        
        self.assertEqual(result.exit_code, 0)
        mock_run.assert_called_once()
        self.assertIn('Successfully submitted job to SLURM.', result.output)

    @patch('subprocess.run')
    def test_run_failure(self, mock_run):
        """Test the run function handles a submission failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr='Failed to submit job')
       
        result = self.runner.invoke(cli, ['run', '-i', self.biom_path, '-m', self.metadata_path, '-f', self.formula, '-o', self.output_dir, '-e', self.email])
        
        self.assertEqual(result.exit_code, 0)
        mock_run.assert_called_once()
        self.assertIn('Failed to submit job', result.output)

if __name__ == '__main__':
    unittest.main()

class TestCLISummarize():
  pass

class TestCLIPlot():
  pass
