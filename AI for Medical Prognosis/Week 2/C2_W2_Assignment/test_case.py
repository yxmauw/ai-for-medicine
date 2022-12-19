import numpy as np
import pandas as pd
#np.random.seed(3)

### ex1
def fraction_rows_missing_test_case():
    df_test = pd.DataFrame({'a':[None, 1, 1, None], 'b':[1, None, 0, 1]})
    
    return df_test
