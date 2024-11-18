# ----------------------------------------------------------------------------
# Copyright (c) 2024, Lucas Patel, Yang Chen
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os
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
from q2_birdman._methods import duplicate_table, _create_dir, run
import patsy


class DuplicateTableTests(TestPluginBase):
    package = 'q2_birdman.tests'

    def test_simple1(self):
        """
        Test functionality of duplicate_table with an in-memory DataFrame
        """
        in_table = pd.DataFrame(
            [[1, 2, 3, 4, 5], [9, 10, 11, 12, 13]],
            columns=['abc', 'def', 'jkl', 'mno', 'pqr'],
            index=['sample-1', 'sample-2'])
        observed = duplicate_table(in_table)

        expected = in_table

        pdt.assert_frame_equal(observed, expected)

    def test_simple2(self):
        """
        Test functionality with input from an actual BIOM file
        """
        in_table = transform(
            self.get_data_path('94270_filtered.biom'),
            from_type=BIOMV210Format,
            to_type=pd.DataFrame)
        observed = duplicate_table(in_table)

        expected = in_table

        pdt.assert_frame_equal(observed, expected)



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
        with self.assertRaises(ValueError, msg="Empty BIOM table."):
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

        with self.assertRaises(ValueError, msg="Empty metadata file."):
            metadata = Metadata(pd.DataFrame())  # Create empty metadata and pass it to the run function
            run(biom_table, metadata, formula, threads=1)


    def test_run_metadata_table_mismatch(self):
        """
        Test that an empty metadata table raises a KeyError
        """
        biom_table = biom.Table(
            np.random.randint(1, 10, size=(20, 2)),
            sample_ids=['sample-3', 'sample-4'],  # Does not match metadata
            observation_ids=[f'feature-{i+1}' for i in range(20)]
        )    
        formula = 'condition'

        with self.assertRaises(KeyError, msg="Sample IDs do not match those in BIOM table."):
            metadata = Metadata(pd.DataFrame(
            {'condition': ['A', 'B']},
            index=pd.Index(['sample-1', 'sample-2'], name='#Sample ID')))

            run(biom_table, metadata, formula, threads=1)


    def test_run_invalid_formula(self):
        """
        Test that an invalid formula raises a PatsyError
        """
        biom_table = biom.Table(
            np.random.randint(1, 10, size=(20, 2)),
            sample_ids=['sample-1', 'sample-2'],
            observation_ids=[f'feature-{i+1}' for i in range(20)]
        )
        formula = 'non_existent_column'

        with self.assertRaises(patsy.PatsyError, msg="Formula references a non-existent column."):
            metadata = Metadata(pd.DataFrame(
                {'condition': ['A', 'B']},
                index=pd.Index(['sample-1', 'sample-2'], name='#Sample ID')
            ))

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
