import numpy as np
import pandas as pd
np.random.seed(3)

### ex2
def hazard_ratio_test_case(i, j, one_hot_train):
    case_1 = one_hot_train.iloc[i, :].drop(['time', 'status'])
    case_2 = one_hot_train.iloc[j, :].drop(['time', 'status'])
    
    return case_1, case_2

### ex3
def harrell_c_test_case():
    y_true_1 = [30, 12, 84, 9]
    
    event_1 = [1, 1, 1, 1]
    scores_1 = [0.5, 0.9, 0.1, 1.0]
    
    scores_2 = [0.9, 0.5, 1.0, 0.1]
    
    event_3 = [1, 0, 1, 1]
    scores_3 = [0.5, 0.9, 0.1, 1.0]
    
    y_true_4 = [30, 30, 20, 20]
    event_4 = [1, 0, 1, 0]
    scores_4 = [10, 5, 15, 20]
    
    y_true_5 = list(reversed([30, 30, 30, 20, 20]))
    event_5 = [0, 1, 0, 1, 0]
    scores_5 = list(reversed([15, 10, 5, 15, 20]))
    
    y_true_6 = [10,10]
    event_6 = [0,1]
    scores_6 = [4,5]
    
    return y_true_1, event_1, scores_1, scores_2, event_3, scores_3, y_true_4, event_4, scores_4, y_true_5, event_5, scores_5, y_true_6, event_6, scores_6

