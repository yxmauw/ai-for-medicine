import numpy as np
import pandas as pd
import seaborn as sns
from test_utils import *
from test_case import *
from IPython.display import display
np.random.seed(3)

### ex1
def to_one_hot_test(target_to_one_hot, df_train, df_val, df_test):
    to_encode = ['spiders', 'stage']
    
    def to_list(target_to_one_hot, to_encode, df_train, df_val, df_test):
        one_hot_train = target_to_one_hot(df_train, to_encode)
        one_hot_val = target_to_one_hot(df_val, to_encode)
        one_hot_test = target_to_one_hot(df_test, to_encode)
        
        return one_hot_train.columns.tolist(), one_hot_val.columns.tolist(), one_hot_test.columns.tolist()
    
    one_hot_train_to_list, one_hot_val_to_list, one_hot_test_to_list = to_list(target_to_one_hot, to_encode, df_train, df_val, df_test)
    
    print("One hot val columns:\n\n", one_hot_val_to_list, "\n")
    print("There are", len(one_hot_val_to_list), "columns\n")
    
    expected_output = (['time', 'status', 'trt', 'age', 'sex', 'ascites', 'hepato', 'edema', 'bili', 
                        'chol', 'albumin', 'copper', 'alk.phos', 'ast', 'trig', 'platelet', 'protime', 'spiders_1.0',
                        'stage_2.0', 'stage_3.0', 'stage_4.0'], 
                       ['time', 'status', 'trt', 'age', 'sex', 'ascites', 'hepato', 'edema', 'bili', 'chol', 'albumin', 
                        'copper', 'alk.phos', 'ast', 'trig', 'platelet', 'protime', 'spiders_1.0', 'stage_2.0', 'stage_3.0',
                        'stage_4.0'], 
                       ['time', 'status', 'trt', 'age', 'sex', 'ascites', 'hepato', 'edema', 'bili', 'chol', 'albumin', 'copper',
                        'alk.phos', 'ast', 'trig', 'platelet', 'protime', 'spiders_1.0', 'stage_2.0', 'stage_3.0', 'stage_4.0'])
        
        
    test_cases = [
        {
            "name":"datatype_check",
            "input": [target_to_one_hot, to_encode, df_train, df_val, df_test],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [target_to_one_hot, to_encode, df_train, df_val, df_test],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [target_to_one_hot, to_encode, df_train, df_val, df_test],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, to_list)
    


##############################################        
### ex2
def hazard_ratio_test(target, i, j, one_hot_train, cph):
    case_1, case_2 = hazard_ratio_test_case(i, j, one_hot_train)
    
    print(target(case_1.values, case_2.values, cph.params_.values), "\n")
    
    expected_output = np.float64(15.029017732492221)
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [case_1.values, case_2.values, cph.params_.values],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [case_1.values, case_2.values, cph.params_.values],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [case_1.values, case_2.values, cph.params_.values],
            "expected": expected_output,
            "error": "Wrong output. One possible solution: make sure i = 1 and j = 5"
        }
    ]
    
    multiple_test(test_cases, target)
    


##############################################        
### ex3
def harrell_c_test(target):
    
    y_true_1, event_1, scores_1, scores_2, event_3, scores_3, y_true_4, event_4, scores_4, y_true_5, event_5, scores_5, y_true_6, event_6, scores_6 = harrell_c_test_case()
    
    print("Test Case 1\n")
    print("y_true: ", y_true_1)
    print("scores: ", scores_1)
    print("event:  ", event_1)
    print("Output: ", target(y_true_1, scores_1, event_1))
    expected_output_1 = 1.0
    
    print("\nTest Case 2\n")
    print("y_true: ", y_true_1)
    print("scores: ", scores_2)
    print("event:  ", event_1)
    print("Output: ", target(y_true_1, scores_2, event_1))
    expected_output_2 = 0.0
    
    print("\nTest Case 3\n")
    print("y_true: ", y_true_1)
    print("scores: ", scores_3)
    print("event:  ", event_3)
    print("Output: ", target(y_true_1, scores_3, event_3))
    expected_output_3 = 1.0
    
    print("\nTest Case 4\n")
    print("y_true: ", y_true_4)
    print("scores: ", scores_4)
    print("event:  ", event_4)
    print("Output: ", target(y_true_4, scores_4, event_4))
    expected_output_4 = 0.75
    
    print("\nTest Case 5\n")
    print("y_true: ", y_true_5)
    print("scores: ", scores_5)
    print("event:  ", event_5)
    print("Output: ", target(y_true_5, scores_5, event_5))
    expected_output_5 = 0.5833333333333334
    
    print("\nTes Case 6\n")
    print("y_true: ", y_true_6)
    print("scores: ", scores_6)
    print("event:  ", event_6)
    print("Output: ", target(y_true_6, scores_6, event_6), "\n")
    expected_output_6 = 1.0
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [y_true_1, scores_1, event_1],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [y_true_1, scores_1, event_1],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [y_true_1, scores_1, event_1],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [y_true_1, scores_2, event_1],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2."
        },
        {
            "name": "equation_output_check",
            "input": [y_true_1, scores_3, event_3],
            "expected": expected_output_3,
            "error": "Wrong output for Test Case 3."
        },
        {
            "name": "equation_output_check",
            "input": [y_true_4, scores_4, event_4],
            "expected": expected_output_4,
            "error": "Wrong output for Test Case 4."
        },
        {
            "name": "equation_output_check",
            "input": [y_true_5, scores_5, event_5],
            "expected": expected_output_5,
            "error": "Wrong output for Test Case 5."
        },
        {
            "name": "equation_output_check",
            "input": [y_true_6, scores_6, event_6],
            "expected": expected_output_6,
            "error": "Wrong output for Test Case 6."
        }
    ]
    
    multiple_test(test_cases, target)
        

