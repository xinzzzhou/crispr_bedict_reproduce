'''
Processs validate/test data & only reserve two column (['ID','Sequence'])
Author: Xin Zhou
Date: 22 Sep, 2021
'''

import pandas as pd
import os

editor = 'Target-AID'
base_dir = "/home/data/bedict_reproduce"
seq_df = pd.read_csv(os.path.join(base_dir, 'data/test_data', editor, 'perbase.csv'), header=0)
# for row in seq_df.iterrows():
# for row in seq_df.iterrows():
#     row[1]
new = seq_df.loc[:, ['ID','Sequence']]
new.to_csv(os.path.join(base_dir, 'data/test_data', editor, 'perbase_testdata_format.csv'), sep=',')