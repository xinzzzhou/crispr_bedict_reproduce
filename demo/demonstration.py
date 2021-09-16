import os
import datetime
import numpy as np
import scipy
import pandas as pd
import torch
from torch import nn

import criscas
from criscas.utilities import create_directory, get_device, report_available_cuda_devices
from criscas.predict_model import *

# base_dir = os.path.abspath('...')
base_dir = "/home/data/bedict"
'''Read sample data'''
seq_df = pd.read_csv(os.path.join(base_dir, 'sample_data', 'abemax_sampledata.csv'), header=0)
# create a directory where we dump the predictions of the models
csv_dir = create_directory(os.path.join(base_dir, 'sample_data', 'predictions'))

'''Specify device (i.e. CPU or GPU) to run the models on'''
report_available_cuda_devices()
# instantiate a device using the only one available :P
device = get_device(True, 0)

'''Create a BE-DICT model by sepcifying the target base editor'''
base_editor = 'ABEmax'
bedict = BEDICT_CriscasModel(base_editor, device)
pred_w_attn_runs_df, proc_df = bedict.predict_from_dataframe(seq_df)

pred_option = 'mean'
pred_w_attn_df = bedict.select_prediction(pred_w_attn_runs_df, pred_option)
pred_w_attn_runs_df.to_csv(os.path.join(csv_dir, f'predictions_allruns.csv'))
pred_w_attn_df.to_csv(os.path.join(csv_dir, f'predictions_predoption_{pred_option}.csv'))

'''Generate attention plots'''
# create a dictionary to specify target sequence and the position we want attention plot for
# we are targeting position 5 in the sequence
seqid_pos_map = {'CTRL_HEKsiteNO1':[5], 'CTRL_HEKsiteNO2':[5]}
pred_option = 'mean'
apply_attn_filter = False
bedict.highlight_attn_per_seq(pred_w_attn_runs_df,
                              proc_df,
                              seqid_pos_map=seqid_pos_map,
                              pred_option=pred_option,
                              apply_attnscore_filter=apply_attn_filter,
                              fig_dir=None)