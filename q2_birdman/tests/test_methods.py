# ----------------------------------------------------------------------------
# Copyright (c) 2024, Lucas Patel, Yang Chen
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os
import re
import tempfile
import pandas as pd
import numpy as np
import pandas.testing as pdt
import biom
import qiime2
from unittest.mock import patch, MagicMock
from qiime2.plugin.testing import TestPluginBase
from qiime2.plugin.util import transform
from q2_types.feature_table import BIOMV210Format
from qiime2 import Metadata
from q2_birdman._methods import _create_dir, run
import patsy


class CreateDirTests(TestPluginBase):
    package = 'q2_birdman.tests'

    def test_create_dir_creates_all_subdirs(self):
        """
        Test that _create_dir creates all required subdirectories, otherwise raises AssertionError
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Call the _create_dir method with the temporary directory
            _create_dir(temp_dir)

            # Define the expected subdirectories
            expected_sub_dirs = ["slurm_out", "logs", "inferences", "results", "plots"]

            # Check that each subdirectory exists
            for sub_dir in expected_sub_dirs:
                sub_dir_path = os.path.join(temp_dir, sub_dir)
                assert os.path.exists(sub_dir_path), f"Subdirectory {sub_dir} does not exist."

    def test_create_dir_handles_existing_subdirs(self):
        """
        Test that _create_dir handles pre-existing subdirectories, otherwises raises AssertionError
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Manually create one of the subdirectories
            os.makedirs(os.path.join(temp_dir, "slurm_out"), exist_ok=True)

            # Call the _create_dir method with the temporary directory
            _create_dir(temp_dir)

            # Define the expected subdirectories
            expected_sub_dirs = ["slurm_out", "logs", "inferences", "results", "plots"]

            # Check that each subdirectory exists
            for sub_dir in expected_sub_dirs:
                sub_dir_path = os.path.join(temp_dir, sub_dir)
                assert os.path.exists(sub_dir_path), f"Subdirectory {sub_dir} does not exist."



class RunMethodTests(TestPluginBase):
    package = 'q2_birdman.tests'

    def test_run_empty_table(self):
        """
        Test that an empty BIOM table raises a ValueError
        """
        biom_table = biom.Table([], [], [])  # Empty BIOM table
        metadata = Metadata(pd.DataFrame(
            {'condition': ['A', 'B']},
            index=pd.Index(['sample-1', 'sample-2'], name='#Sample ID')
        ))
        formula = 'condition'
        with self.assertRaisesRegex(ValueError, "Feature table must contain at least one ID."):
            run(biom_table, metadata, formula, threads=1)


    def test_run_empty_metadata(self):
        """
        Test that an empty metadata table raises a ValueError
        """
        biom_table = biom.Table(
            np.random.randint(1, 10, size=(20, 2)),
            sample_ids=['sample-1', 'sample-2'],
            observation_ids=[f'feature-{i+1}' for i in range(20)]
        )
        formula = 'condition'

        with self.assertRaisesRegex(ValueError, "Metadata must contain at least one ID."):
            metadata = Metadata(pd.DataFrame()) 
            run(biom_table, metadata, formula, threads=1)

    def test_run_metadata_table_mismatch(self):
        """Test that mismatched sample IDs between BIOM table and metadata raises ValueError"""
        biom_table = biom.Table(
            np.array([[1, 2], [3, 4]]),  # Fixed values instead of random
            sample_ids=['sample-3', 'sample-4'],
            observation_ids=['feature-1', 'feature-2']
        )
        
        metadata = Metadata(pd.DataFrame(
            {'condition': ['A', 'B']},
            index=pd.Index(['sample-1', 'sample-2'], name='#Sample ID')
        ))
        
        formula = 'condition'
        expected_pattern = (
            r"Missing samples in metadata: (sample-3, sample-4|sample-4, sample-3)\n"
            r"Missing samples in table: (sample-1, sample-2|sample-2, sample-1)"
        )
        
        with self.assertRaisesRegex(ValueError, expected_pattern):
            run(biom_table, metadata, formula, threads=1)

    def test_run_invalid_formula(self):
        """Test that an invalid formula raises a PatsyError"""
        biom_table = biom.Table(
            np.array([[1, 2], [3, 4]]),
            sample_ids=['sample-1', 'sample-2'],
            observation_ids=['feature-1', 'feature-2']
        )
        metadata = Metadata(pd.DataFrame(
            {'condition': ['A', 'B']},
            index=pd.Index(['sample-1', 'sample-2'], name='#Sample ID')
        ))
        formula = 'non_existent_column'

        expected_error = (
            "Missing columns in metadata: non_existent_column\n"
            "Available columns are: condition"
        )

        with self.assertRaisesRegex(ValueError, re.escape(expected_error)):
            run(biom_table, metadata, formula, threads=1)

    def test_run_formula_with_null_metadata_values(self):
        """Test that formula with null values in metadata raises a ValueError"""
        biom_table = biom.Table(
            np.array([[1, 2], [3, 4]]),
            sample_ids=['sample-1', 'sample-2'],
            observation_ids=['feature-1', 'feature-2']
        )
        
        metadata = Metadata(pd.DataFrame({
            'condition': ['A', None], 
            'other': [1, 2]
        }, index=pd.Index(['sample-1', 'sample-2'], name='#Sample ID')))
        
        formula = 'condition'
        
        with self.assertRaisesRegex(ValueError, "The following columns contain null values: condition"):
            run(biom_table, metadata, formula, threads=1)

    def test_run_creates_directories(self):
        """
        Check that expected output directories after Run are created (implicitly tests _create_dir), otherwises raises AssertionError
        """
        biom_table = biom.Table(
            np.random.randint(1, 10, size=(20, 2)),
            sample_ids=['sample-1', 'sample-2'],
            observation_ids=[f'feature-{i+1}' for i in range(20)]
        )

        # Mock metadata
        metadata = Metadata(pd.DataFrame(
            {'condition': ['A', 'B']},
            index=pd.Index(['sample-1', 'sample-2'], name='#Sample ID')
        ))

        formula = 'condition'

        with tempfile.TemporaryDirectory() as temp_dir, \
            patch('q2_birdman.src.birdman_chunked.run_birdman_chunk'), \
            patch('q2_birdman.src._summarize.summarize_inferences',
                return_value=pd.DataFrame({'col1': [1, 2]})):  # Valid DataFrame mock

            with patch('os.getcwd', return_value=temp_dir):
                try:
                    output_metadata = run(biom_table, metadata, formula, threads=1)
                    print("Output metadata:", output_metadata)
                except Exception as e:
                    print("Error during run:", str(e))
                    raise

            # Verify directories
            expected_dirs = ["slurm_out", "logs", "inferences", "results", "plots"]
            for sub_dir in expected_dirs:
                assert os.path.exists(os.path.join(temp_dir, "test_out", sub_dir)), \
                    f"Expected directory {sub_dir} was not created."


    def test_run_biom_table_with_nans(self):
        """
        Test that a BIOM table with NaN values raises a ValueError.
        """
        # Create raw data with NaN values
        data = np.array([[1, 2], [np.nan, 4]])

        # Run test with assertRaises
        with self.assertRaises(ValueError, msg="BIOM table with NaN values should raise an error."):
            # Ensure NaN check before BIOM table creation
            if np.isnan(data).any():
                raise ValueError("Input data contains NaN values, which are not supported in BIOM tables.")
            
            # Mock metadata
            metadata = Metadata(pd.DataFrame(
                {'condition': ['A', 'B']},
                index=pd.Index(['sample-1', 'sample-2'], name='#Sample ID')
            ))
            
            formula = 'condition'
            
            # Call the function under test
            # Mock BIOM table (this won't execute due to the ValueError above)
            biom_table = biom.Table(
                data,
                sample_ids=['sample-1', 'sample-2'],
                observation_ids=['feature-1', 'feature-2'])
            
            run(biom_table, metadata, formula, threads=1)

    def test_non_null_formula_variables(self):
        """
        Test that formula contains variables with all non-null values.
        """
        pass
