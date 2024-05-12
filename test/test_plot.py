import os # TODO: remoev when setup.py is created
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from io import StringIO
from src._plot import _read_results, _unpack_hdi_and_filter, birdman_plot_multiple_vars


class TestPlot(unittest.TestCase):
    def setUp(self):
        self.inference_data = """Feature\tIntercept_mean
G000714935\t-11.650730
G000185445\t-13.562568
G000014825\t-11.589558
G002020915\t-14.179677
G900129965\t-13.405741
"""
        self.inference_data_df = pd.read_csv(StringIO(self.inference_data), sep='\t', index_col='Feature')

        self.feature_md_data = """Feature\tName
G000714935\tmicrobe1
G000185445\tmicrobe2
G000014825\tmicrobe3
G002020915\tmicrobe4
G900129965\tmicrobe5
"""
        self.feature_md_data_df = pd.read_csv(StringIO(self.feature_md_data), sep='\t', index_col='Feature')

        self.inference_dataframe = pd.DataFrame(
            {
                "Feature": [
                    "G000714935",
                    "G000185445",
                    "G000014825",
                    "G002020915",
                    "G900129965",
                ],
                "Intercept_mean": [
                    -11.650730,
                    -13.562568,
                    -11.589558,
                    -14.179677,
                    -13.405741,
                ],
            }
        )
        self.inference_dataframe.set_index("Feature", drop=True, inplace=True)

        self.feature_md_dataframe = pd.DataFrame(
            {
                "Feature": [
                    "G000714935",
                    "G000185445",
                    "G000014825",
                    "G002020915",
                    "G900129965",
                ],
                "Name": ["microbe1", "microbe2", "microbe3", "microbe4", "microbe5"],
            }
        )
        self.feature_md_dataframe.set_index("Feature", drop=True, inplace=True)
        self.summarized_inference_file = "./data/191733_none.beta_var.tsv"

    def assertIsFile(self, path):
        if not Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))


    @patch('pandas.read_csv')
    def test_read_results_without_feature_metadata(self, mock_read_csv):
        mock_read_csv.return_value = self.inference_data_df
        actual = _read_results("path/to/summarized_inferences.tsv", None)
        expected =  self.inference_dataframe
        pd.testing.assert_frame_equal(actual, expected)


    @patch('pandas.read_csv')
    def test_read_results_with_feature_metadata(self, mock_read_csv):
        def side_effect(filepath, sep='\t', index_col="Feature"):
            if 'feature_md' in filepath:
                return self.feature_md_data_df
            else:
                return self.inference_data_df
        mock_read_csv.side_effect = side_effect

        actual = _read_results("path/to.inference.tsv", "path/to/feature_md.tsv")
        expected = pd.DataFrame(
            {
                "Feature": ["microbe1", "microbe2", "microbe3", "microbe4", "microbe5"],
                "Intercept_mean": [
                    -11.650730,
                    -13.562568,
                    -11.589558,
                    -14.179677,
                    -13.405741,
                ],
            }
        )
        expected.set_index("Feature", drop=True, inplace=True)
        pd.testing.assert_frame_equal(actual, expected)


    def test_unpack_hdi_and_filter(self):
        data = {
            'Feature': ['A', 'B'],
            'some_hdi': ['(1.5,3.5)', '(-2.2,0.5)'],
            'some_mean': [2.0, -1.5]
        }
        df = pd.DataFrame(data)
        df.set_index('Feature', inplace=True)
        
        expected_data = {
            'lower': [0.5, 0.7],
            'upper': [1.5, 2.0],
            'credible': ['yes', 'no']
        }
        expected_df = pd.DataFrame(expected_data, index=['A', 'B'])
        expected_df.index.name = 'Feature'
        
        result_df = _unpack_hdi_and_filter(df, 'some_hdi')
        
        np.testing.assert_array_almost_equal(result_df['lower'].values, expected_df['lower'].values)
        np.testing.assert_array_almost_equal(result_df['upper'].values, expected_df['upper'].values)
        pd.testing.assert_series_equal(result_df['credible'], expected_df['credible'])


    def test_birdman_plot_multiple_vars(self):
        expected_files = ["./data/host_age[T.18]_plot.png", "./data/host_age[T.34]_plot.png"]
        for file in expected_files:
            if Path(file).resolve().is_file():
                Path.unlink(Path(file))
        birdman_plot_multiple_vars(self.summarized_inference_file, "./data/", None, "host_age[T.34],host_age[T.18]")
        for file in expected_files:
            self.assertIsFile(file)

if __name__ == "__main__":
    unittest.main()
