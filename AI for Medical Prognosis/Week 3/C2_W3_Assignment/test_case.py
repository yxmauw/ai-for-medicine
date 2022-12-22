import numpy as np
import pandas as pd
np.random.seed(3)

### ex2
def naive_estimator_test_case():
    sample_df_1 = pd.DataFrame(columns = ["Time", "Event"])
    sample_df_1.Time = [5, 10, 15]
    sample_df_1.Event = [0, 1, 0]
    
    sample_df_2 = pd.DataFrame({'Time': [5,5,10],
                                'Event': [0,1,0]
                               })
    
    return sample_df_1, sample_df_2

### ex3
def HomemadeKM_test_case():
    sample_df_1 = pd.DataFrame(columns = ["Time", "Event"])
    sample_df_1.Time = [5, 10, 15]
    sample_df_1.Event = [0, 1, 0]
    
    sample_df_2 = pd.DataFrame(columns = ["Time", "Event"])
    sample_df_2.loc[:, "Time"] = [2, 15, 12, 10, 20]
    sample_df_2.loc[:, "Event"] = [0, 0, 1, 1, 1]
    
    return sample_df_1, sample_df_2