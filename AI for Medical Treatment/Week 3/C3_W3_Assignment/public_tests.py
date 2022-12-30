import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from test_utils import *
from test_cases import *

### ex1
def grad_cam_test(target, im_path, mean, std, load_image_normalize, model, reference):
    
    im = grad_cam_test_case(target, im_path, mean, std, load_image_normalize, model)
    
    cam = target(model, im, 5, 'conv5_block16_concat') # Mass is class 5
    
    # Loads reference CAM to compare our implementation with.
    error = np.mean((cam - reference)**2)
    
    print("Error from reference should be less than 0.05")
    print("Your error from reference: ", error, "\n")
    
    expected_value = True
    
    ### to check output value is less than 0.05
    def target_value_test(error):
        if error < 0.05:
            return True
        else:
            return False
        
    ### to check output value is less than 0.05
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [error],
            "expected": expected_value,
            "error": "Wrong output. Your reference is greater than 0.05"
        }
    ]
    
    multiple_test(test_cases, target_value_test)
    
    return cam


### ex3
def permute_feature_test(target):
    
    example_df = permute_feature_test_case()
    
    print("Test Case\n")
    print("Original dataframe:\n")
    print(example_df, "\n")
    
    print("col1 permuted:\n")
    print(target(example_df, 'col1'), "\n")
    
    print("Average values after computing over 1000 runs:")
        
    def test_target_values(target):
        col1_values = np.zeros((3, 1000))

        np.random.seed(0) # Adding a constant seed so we can always expect the same values and evaluate correctly. 

        for i in range(1000):
            col1_values[:, i] = target(example_df, 'col1')['col1'].values
        
        return np.mean(col1_values, axis=1)
    
    expected_output = np.array([0.976, 1.03,  0.994])
        
    print("Average of col1: {}".format(test_target_values(target)))
    print("\n")
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [target],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [target],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [target],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, test_target_values)

## ex4
def permutation_importance_test(target, cindex):
    print("Test Case\n")
    print("You check your answers on a Logistic Regression on a dataset")
    print("where y is given by a sigmoid applied to the important feature.") 
    print("The unimportant feature is random noise.")
    print("\n")
    
    example_df, example_y, example_model = permutation_importance_test_case()
    
    num_samples = 100
    example_importances = target(example_df, example_y, example_model, cindex, num_samples)
    
    ### to check the output's shape and data type
    expected_shape_type = pd.DataFrame({"important": 0.5, "unimportant": 0.0}, index=['importance'])
    expected_flag = True
    
    def target_output_value_test(example_importances):
        important = example_importances.iloc[0]['important']
        unimportant = example_importances.iloc[0]['unimportant']
        flag = np.allclose([0.5, 0.0], [important, unimportant], rtol=1e-01, atol=1e-02)
        
        return flag
    
    print("Computed Importances:")
    print(example_importances)
    print("\n")
        
    ### for datatype and shape check
    test_cases = [
        {
            "name":"datatype_check",
            "input": [example_df, example_y, example_model, cindex, num_samples],
            "expected": expected_shape_type,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [example_df, example_y, example_model, cindex, num_samples],
            "expected": expected_shape_type,
            "error": "Wrong shape."
        }
    ]
    
    multiple_test(test_cases, target)
    
    ### for checking closeness of expected values
    test_cases = [
        {
            "name":"equation_output_check",
            "input": [example_importances],
            "expected": expected_flag,
            "error": "Wrong output. Your values are not closer to expected 0.5 and 0.0"
        }
    ]
    
    multiple_test(test_cases, target_output_value_test)
    
