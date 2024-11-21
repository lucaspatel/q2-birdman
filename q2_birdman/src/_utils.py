import patsy
import biom
import pandas as pd
import click
from pathlib import Path

def validate_table_and_metadata(table, metadata):
    """
    Validates BIOM table and QIIME2 metadata compatibility.
    
    Parameters
    ----------
    table : biom.Table
        BIOM table to validate
    metadata : qiime2.Metadata
        QIIME2 metadata to validate
        
    Returns
    -------
    bool
        True if valid, raises ValueError otherwise
        
    Raises
    ------
    ValueError
        If validation fails
    """
    table_ids = set(table.ids(axis='sample'))
    metadata_ids = set(metadata.to_dataframe().index)
    
    if not table_ids:
        raise ValueError("Feature table must contain at least one ID.")
        
    if not metadata_ids:
        raise ValueError("Metadata must contain at least one ID.")
        
    if table_ids != metadata_ids:
        missing_from_metadata = table_ids - metadata_ids
        missing_from_table = metadata_ids - table_ids
        error_msg = []
        if missing_from_metadata:
            error_msg.append(f"Missing samples in metadata: {', '.join(missing_from_metadata)}")
        if missing_from_table:
            error_msg.append(f"Missing samples in table: {', '.join(missing_from_table)}")
        raise ValueError('\n'.join(error_msg))
    
    return True

def validate_formula(formula, table, metadata):
    """
    Validates a Patsy formula against available metadata columns.
    """
    metadata = metadata.to_dataframe()
    sample_names = table.ids(axis="sample")
    available_columns = set(str(col) for col in metadata.columns)  # Force strings

    try:
        design_info = patsy.ModelDesc.from_formula(formula)
        term_names = set()
        for term in design_info.rhs_termlist:
            for factor in term.factors:
                if hasattr(factor, 'name'):
                    term_names.add(str(factor.name()))  # Force string

        # Check if all required terms exist in metadata
        missing_terms = term_names - available_columns
        if missing_terms:
            raise ValueError(f"Missing columns in metadata: {', '.join(str(x) for x in missing_terms)}\n"
                           f"Available columns are: {', '.join(str(x) for x in available_columns)}")

        # Check for null values in required columns
        null_columns = [
            str(term) for term in term_names 
            if pd.isna(metadata[term]).any()  # Use pandas null check
        ]
                
        if null_columns:
            raise ValueError(f"The following columns contain null values: {', '.join(null_columns)}")

        # Try to actually build the design matrix as final validation
        patsy.dmatrix(formula, metadata.loc[sample_names], return_type="dataframe")
        return True

    except patsy.PatsyError as e:
        raise ValueError(f"Invalid Patsy formula: {str(e)}\n"
                        f"Available columns are: {', '.join(str(x) for x in available_columns)}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error validating formula: {str(e)}")

def _create_folder_without_clear(dir):
    dir = Path(dir)
    if dir.exists() and dir.is_dir():
        return
    else:
        dir.mkdir(parents=True, exist_ok=True)
