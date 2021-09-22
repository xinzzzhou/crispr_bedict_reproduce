'''
Able to process more data
Author: Xin Zhou
Date: 22 Sep, 2021
'''

import argparse
from criscas.utilities import create_directory, get_device, report_available_cuda_devices
from criscas.predict_model import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train crispr.")
    parser.add_argument('--editor', type=str, default="CBE4max",
                        help='Input editor. Selection: ABEmax, ABE8e, CBE4max, Target-AID')
    parser.add_argument('--base_dir', type=str, default="/home/data/bedict_reproduce",
                        help='path to the project.')
    parser.add_argument('--pred_option', type=str, default="mean",
                        help='Way to compute prediction score. Option: mean, median, min, max')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    '''Specify device (i.e. CPU or GPU) to run the models on'''
    report_available_cuda_devices()
    device = get_device(True, 0)

    '''Create a BE-DICT model by sepcifying the target base editor'''
    bedict = BEDICT_CriscasModel(args.editor, device)
    # load data
    seq_df = pd.read_csv(os.path.join(args.base_dir, 'data/test_data', args.editor, 'perbase_testdata_format.csv'), header=0).iloc[:,1:]
    pred_w_attn_runs_df, proc_df = bedict.predict_from_dataframe(seq_df)
    # merge 5 runs result
    pred_w_attn_df = bedict.select_prediction(pred_w_attn_runs_df, args.pred_option)
    # record the result

    csv_dir = create_directory(os.path.join(args.base_dir, 'data/test_data',  args.editor, 'predictions'))
    pred_w_attn_runs_df.to_csv(os.path.join(csv_dir, f'predictions_allruns.csv'))
    pred_w_attn_df.to_csv(os.path.join(csv_dir, f'predictions_predoption_{args.pred_option}.csv'))
