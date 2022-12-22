import numpy as np
import pandas as pd
import seaborn as sns
from test_utils import *
from test_case import *
from IPython.display import display
np.random.seed(3)

### ex1
def frac_censored_test(target, data):
    data = data
    print("Observations which were censored: ", target(data))
    expected_output = 0.325
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [data],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [data],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [data],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)
    


##############################################        
### ex2
def naive_estimator_test(target):
    sample_df_1, sample_df_2 = naive_estimator_test_case()
    
    print("Sample 1 dataframe for testing code:\n")
    print(sample_df_1)
    print("\n")
    
    print("Test Case 1: S(3)")
    print("Output: ", target(3, sample_df_1))

    print("\nTest Case 2: S(12)")
    print("Output: ", target(12, sample_df_1))

    print("\nTest Case 3: S(20)")
    print("Output: ", target(20, sample_df_1))
    
    print("\nSample 2 dataframe for testing code:\n")
    print("\n", sample_df_2, "\n")

    print("Test case 4: S(5)")
    print("Output: ", target(5, sample_df_2), "\n")
    
    expected_output_1 = 1.0
    expected_output_2 = 0.5
    expected_output_3 = 0.0
    expected_output_4 = 0.5
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [3, sample_df_1],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [3, sample_df_1],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [3, sample_df_1],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [12, sample_df_1],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2."
        },
        {
            "name": "equation_output_check",
            "input": [20, sample_df_1],
            "expected": expected_output_3,
            "error": "Wrong output for Test Case 3."
        },
        {
            "name": "equation_output_check",
            "input": [5, sample_df_2],
            "expected": expected_output_4,
            "error": "Wrong output for Test Case 4."
        }
    ]
    
    multiple_test(test_cases, target)
    


##############################################        
### ex3
def HomemadeKM_test(target):
    
    sample_df_1, sample_df_2 = HomemadeKM_test_case()
    
    print("Test Case 1\n")
    print(sample_df_1.head(), "\n")
    x, y = target(sample_df_1)
    print("Test Case 1 Event times: {}, Survival Probabilities: {}".format(x, y))
    
    print("\nTest Case 2\n")
    print(sample_df_2.head(), "\n")
    x, y = target(sample_df_2)
    print("Test Case 2 Event times: {}, Survival Probabilities: {}".format(x, y), "\n")
    
    expected_output_1 = (np.array([0, 5, 10, 15]), np.array([1.0, 1.0, 0.5, 0.5]))
    expected_output_2 = (np.array([0, 2, 10, 12, 15, 20]), np.array([1.0, 1.0, 0.75, 0.5, 0.5, 0.0]))
    
    test_cases = [
        
        {
            "name": "shape_check",
            "input": [sample_df_1],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [sample_df_1],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [sample_df_2],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2."
        }
    ]
    
    multiple_test(test_cases, target)
    
    
    