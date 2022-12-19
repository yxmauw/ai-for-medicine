import numpy as np
import pandas as pd
import seaborn as sns
from test_utils import *
from test_case import *
from IPython.display import display
#np.random.seed(3)

### ex1
def fraction_rows_missing_test(target, X_train, X_val, X_test):
    df_test = fraction_rows_missing_test_case()
    
    print("Example dataframe:\n\n", df_test, "\n")
    print("Computed fraction missing: ", target(df_test))
    print("Fraction of rows missing from X_train: ", target(X_train))
    print("Fraction of rows missing from X_val: ", target(X_val))
    print("Fraction of rows missing from X_test: ", target(X_test))
    
    expected_output = 0.75
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [df_test],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [df_test],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [df_test],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)
