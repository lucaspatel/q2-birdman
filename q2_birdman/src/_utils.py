import patsy
import biom
import pandas as pd
import click
from pathlib import Path

def is_valid_patsy_formula(formula, table, metadata):
    """
    Validates whether a string is a valid Patsy-style formula by attempting to construct a design matrix using data from specified paths.

    Parameters:
    - formula (str): The Patsy formula string to validate.
    - table_path (str): The file path to the BIOM table, which provides sample IDs.
    - metadata_path (str): The file path to the metadata CSV file, which contains variables expected in the formula as column headers.

    Returns:
    - bool: True if the formula is valid within the context of the provided data, False otherwise.

    This function attempts to construct a design matrix using the formula and data. It first extracts variable names from the metadata CSV file headers. Then, it loads sample IDs from the BIOM table and uses these to align and subset the metadata. An exception in parsing or matrix construction results in a False return, and the error is printed along with a list of valid variables.
    """
    sample_names = table.ids(axis="sample")
    print(metadata)
    variables = list(metadata.columns)
    formatted_list = ', '.join(variables)
    try:
        patsy.dmatrix(formula, metadata.loc[sample_names], return_type="dataframe")
        return True
    except Exception as e:
        click.echo(f"Invalid formula: {e}.", err=True)
        click.echo(f"Valid variables are: {formatted_list}", err=True)
        return False
    

def _create_folder_without_clear(dir):
    dir = Path(dir)
    if dir.exists() and dir.is_dir():
        return
    else:
        dir.mkdir(parents=True, exist_ok=True)
