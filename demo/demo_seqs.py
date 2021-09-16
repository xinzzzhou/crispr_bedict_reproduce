import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import haplotype
from haplotype.dataset import *
from haplotype.data_preprocess import *
from haplotype.utilities import *
from haplotype.predict_model import BEDICT_HaplotypeModel

curr_pth = os.path.abspath('../')

'''Read sample data'''
df = pd.read_csv(f"{curr_pth}/sample_data/bystander_sampledata.csv")

# create a directory where we dump the predictions of the models
csv_dir = create_directory(f"{curr_pth}/sample_data/predictions_haplo")

'''Specify device (i.e. CPU or GPU) to run the models on'''
report_available_cuda_devices()
# instantiate a device using the only one available :P
device = get_device(True, 0)

'''Create a BE-DICT (bystander) model'''
for gr_name, gr_df in df.groupby(by=['Editor']):
    print(gr_name)
    # display(gr_df)
seqconfig_dataproc = SeqProcessConfig(20, (1,20), (3,10), 1)

teditor = 'ABEmax'
cnv_nucl = ('A', 'G')
seq_processor = HaplotypeSeqProcessor(teditor, cnv_nucl, seqconfig_dataproc)

seqconfig_datatensgen = SeqProcessConfig(20, (1,20), (1 ,20), 1)
bedict_haplo = BEDICT_HaplotypeModel(seq_processor, seqconfig_datatensgen, device)

'''Run a BE-DICT (bystander) model'''
sample_df = df.loc[df['Editor'] == 'ABEmax'].copy()

'''Main (default) mode'''
# The are different modes (i.e. options) to run the model. In the main (default) mode, we just provide the data frame that includes the input sequences and the model will generate the different outcome sequences and their corresponding probability.
dloader = bedict_haplo.prepare_data(sample_df,
                                    ['seq_id','Inp_seq'],
                                    outpseq_col=None,
                                    outcome_col=None,
                                    renormalize=False,
                                    batch_size=500)
# After we parse the data, we can run the model to get predictions. We have trained 5 different models for every base editor (denoted by runs) which we will use on this data and take their average predictions. The trained bystander/haplotype models are found under the trained_models/bystander directory.
num_runs = 5  # number of models
pred_df_lst = []  # list to hold the prediction data frames for each model run
for run_num in range(num_runs):
    # specify model directory based on the chose editor
    model_dir = f"{curr_pth}/trained_models/bystander/{teditor}/train_val/run_{run_num}"
    print('run_num:', run_num)
    print('model_dir:', model_dir)

    pred_df = bedict_haplo.predict_from_dloader(dloader,
                                                model_dir,
                                                outcome_col=None)
    pred_df['run_num'] = run_num
    pred_df_lst.append(pred_df)
# aggregate predictions in one data frame
pred_df_unif = pd.concat(pred_df_lst, axis=0, ignore_index=True)
# sanity check
check_na(pred_df_unif)
# compute average predictions
agg_df = bedict_haplo.compute_avg_predictions(pred_df_unif)
# sanity check
check_na(agg_df)
print(agg_df)

'''Visualize results'''
#
from haplotype.data_preprocess import get_char
from IPython.core.display import HTML
from IPython.core import display
from haplotype.data_preprocess import VizInpOutp_Haplotype
tseqids = sample_df['seq_id']
res_html = bedict_haplo.visualize_haplotype(agg_df,
                                            tseqids,
                                            ['seq_id','Inp_seq'],
                                            'Outp_seq',
                                            'pred_score',
                                            predscore_thr=0.)
# for seq_id in res_html:
#     display(HTML(res_html[seq_id]))

# save the visualization of each sequence (or all sequences) in html format
from haplotype.data_preprocess import HaplotypeVizFile
vf = HaplotypeVizFile(os.path.join(curr_pth, 'haplotype', 'viz_resources'))
for seq_id in res_html:
    # display(HTML(res_html[seq_id]))
    vf.create(res_html[seq_id], csv_dir, f'{teditor}_{seq_id}_haplotype')