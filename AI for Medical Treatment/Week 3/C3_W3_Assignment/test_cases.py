import numpy as np
import pandas as pd
import sklearn

### ex1
def grad_cam_test_case(target, im_path, mean, std, load_image_normalize, model):
        im = load_image_normalize(im_path, mean, std)
        
        return im

### ex3
def permute_feature_test_case():
    example_df = pd.DataFrame({'col1': [0, 1, 2], 'col2':['A', 'B', 'C']})
    
    return example_df

### ex4
def permutation_importance_test_case():
    example_df = pd.DataFrame({'important': np.random.normal(size=(1000)), 'unimportant':np.random.normal(size=(1000))})
    example_y = np.round(1 / (1 + np.exp(-example_df.important)))
    example_model = sklearn.linear_model.LogisticRegression(fit_intercept=False).fit(example_df, example_y)
    
    return example_df, example_y, example_model

