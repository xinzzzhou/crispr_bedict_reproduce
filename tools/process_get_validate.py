'''
Processs data & get validate/test data
Author: Xin Zhou
Date: 22 Sep, 2021
'''
import os
import pandas as pd
import csv

editor='Target-AID'
f = open('/home/data/bedict_reproduce/data/test_data/'+editor+'/perbase_testdata.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)

csv_writer.writerow(["ID","Purpose","Count","Sequence","Position_1","Position_2","Position_3","Position_4","Position_5","Position_6","Position_7","Position_8","Position_9","Position_10","Position_11","Position_12","Position_13","Position_14","Position_15","Position_16","Position_17","Position_18","Position_19","Position_20"])

base_dir = "/home/data/bedict_reproduce"
seq_df = pd.read_csv(os.path.join(base_dir, 'data/test_data', editor, 'perbase.csv'), header=0)
for row in seq_df.iterrows():
    if row[1]['Purpose'] == "Validation/Test":
        csv_writer.writerow(row[1])
f.close()