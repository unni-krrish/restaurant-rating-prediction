import os
import zipfile
import requests

import numpy as np
import pandas as pd


curr_dir = os.path.abspath('')
basedf = pd.read_csv(os.path.join(curr_dir, 'data', 'zomato.csv'))

# rands = np.random.choice(basedf.index, size=5000, replace=False)
# filtered = basedf.iloc[rands, :].reset_index(drop=True)

basedf = basedf[basedf['rate'] != '-']
new_rest_df = basedf[basedf['rate'] == 'NEW'].reset_index(drop=True)

print(new_rest_df.shape)

new_rest_df.to_csv(os.path.join(
    curr_dir, 'data', "test_set_1.csv"), index=False)
