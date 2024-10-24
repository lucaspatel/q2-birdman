# ----------------------------------------------------------------------------
# Copyright (c) 2024, Lucas Patel.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import qiime2 # added
import os
import tempfile
import pandas as pd
from joblib import Parallel, delayed
from qiime2 import Metadata
import biom

from .src.birdman_chunked import run_birdman_chunk
from .src._utils import is_valid_patsy_formula
from .src._summarize import summarize_inferences

def _create_dir(output_dir):
  sub_dirs = ["slurm_out", "logs", "inferences", "results", "plots"]
  for sub_dir in sub_dirs:
      os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)

def run(table: biom.Table, metadata: Metadata, formula: str, threads: int = 16) -> qiime2.Metadata:
#def run(table: qiime2.Artifact, metadata: qiime2.Artifact, formula: str, threads: int = 16) -> qiime2.Metadata:
    """Run BIRDMAn and return the inference results as ImmutableMetadata."""
    
    # Convert QIIME 2 artifact to a BIOM table
    #table = table.view(biom.Table) # added

    # Convert QIIME 2 Metadata to a pandas DataFrame
    metadata_df = metadata.to_dataframe()
    #metadata_df = metadata.view(pd.DataFrame) # added

    # Validate the Patsy formula with the BIOM table and metadata
    is_valid_patsy_formula(formula, table, metadata_df)
    
    # Set the number of chunks for parallel processing
    chunks = 20

    output_dir = os.path.join(os.getcwd(), "test_out") 
    _create_dir(output_dir)
    print(f"Output dir is {output_dir}")
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    def run_chunk(chunk_num):
        log_path = os.path.join(output_dir, "logs", f"chunk_{chunk_num}.log")
        run_birdman_chunk(
            table=table,
            metadata=metadata_df,
            formula=formula,
            inference_dir=output_dir,
            num_chunks=chunks,
            chunk_num=chunk_num,
            logfile=log_path
        )

    Parallel(n_jobs=threads)(
        delayed(run_chunk)(i) for i in range(1, chunks + 1)
    )

    summarized_results = summarize_inferences(output_dir)
    output_metadata = Metadata(summarized_results)

    print(f"Results are stored in: {output_dir}")
    return output_metadata
