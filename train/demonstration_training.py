'''
Able to process more data
Author: Xin Zhou
Date: 22 Sep, 2021
'''

import argparse
from criscas.utilities import create_directory, get_device, report_available_cuda_devices
from criscas.predict_model_training import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train crispr.")
    parser.add_argument('--editor', type=str, default="ABEmax",
                        help='Input editor. Selection: ABEmax, ABE8e, CBE4max, Target-AID')
    parser.add_argument('--base_dir', type=str, default="/home/data/bedict_reproduce",
                        help='path to the project.')
    parser.add_argument('--pred_option', type=str, default="mean",
                        help='Way to compute prediction score. Option: mean, median, min, max')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size of the data')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1')
    parser.add_argument('--beta2', type=float, default=0.98,
                        help='Beta2')
    parser.add_argument('--eps', type=float, default=1e-08,
                        help='Eps for training the model')
    parser.add_argument('--wdecay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Weight decay')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    '''Specify device (i.e. CPU or GPU) to run the models on'''
    report_available_cuda_devices()
    device = get_device(True, 3)


    '''Create a BE-DICT model by sepcifying the target base editor'''
    model = BEDICT_CriscasModel(args.editor, device)

    # criteria = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
    #                              weight_decay=args.wdecay, amsgrad=False)


    '''Load data'''
    # seq_df = pd.read_csv(os.path.join(args.base_dir, 'data/test_data', args.editor, 'perbase_testdata_format.csv'),
    seq_df = pd.read_csv(os.path.join(args.base_dir, 'data/test_data', args.editor, 'perbase_testdata_train_format.csv'),
                         header=0).iloc[:, 1:]

    y = model(seq_df, args.batch_size, args.lr, args.beta1, args.beta2, args.eps, args.wdecay, args.threshold)
    '''train the data'''
    # model.train()
    # for epoch in range(args.epochs):
        # for tra_i, batch in enumerate(train_loader):
        #     pass


    '''Make prediction'''
    pred_w_attn_runs_df, proc_df = model.predict_from_dataframe(seq_df, args.threshold)


    '''Merge the results together'''
    # merge 5 runs result
    pred_w_attn_df = model.select_prediction(pred_w_attn_runs_df, args.pred_option)
    pred = pred_w_attn_df['prob_score_class1']


    '''Record the result'''
    csv_dir = create_directory(os.path.join(args.base_dir, 'data/test_data',  args.editor, 'predictions'))
    pred_w_attn_runs_df.to_csv(os.path.join(csv_dir, f'predictions_allruns.csv'))
    pred_w_attn_df.to_csv(os.path.join(csv_dir, f'predictions_predoption_{args.pred_option}.csv'))
