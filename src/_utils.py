import patsy

def is_valid_patsy_formula(formula, data={}):
    """
    Check if the given formula is a valid Patsy-style formula.

    Parameters:
        formula (str): The formula string to validate.
        data (dict, optional): A dictionary representing a sample data frame where
                               keys are column names and values are lists of data.
                               Defaults to an empty dictionary, which will not validate
                               variable names in the formula.

    Returns:
        bool: True if the formula is valid, False otherwise.
    """
    try:
        # Attempt to create a design matrix. If no data is provided,
        # Patsy will only check for syntax correctness.
        patsy.dmatrix(formula, data)
        return True
    except (patsy.PatsyError, ValueError) as e:
        # Print the error message optionally or handle it based on your needs
        print(f"Invalid formula: {e}")
        return False

# Example usage
print(is_valid_patsy_formula("y ~ x1 + x2"))  # Should return True if syntax is correct

