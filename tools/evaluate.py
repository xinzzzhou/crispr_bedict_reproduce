import argparse
# import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

def findA(seq):
    idx = 0
    position = []
    for s in seq:
        idx += 1
        if s == 'A':
            position.append(idx)
    return position

def get_value(seqids, pred_reader, true_reader, label_threshold):
    dict_true_pred = defaultdict(list)

    for row in true_reader:
        if row['ID'] in seqids:
            Apos = findA(row['Sequence'])
            for idx in Apos:
                true_freq = row['Position_'+str(idx)]
                key = row['ID'] + "|" + str(idx)
                if float(true_freq) >= label_threshold:
                    dict_true_pred[key].append(1)
                else:
                    dict_true_pred[key].append(0)

    for row in pred_reader:
        if row['id'] in seqids:
            key = row['id'] + "|" + str(int(row['base_pos'])+1)
            dict_true_pred[key].append(float(row['prob_score_class1']))

    true, pred = [], []
    for k,v in dict_true_pred.items():
        true.append(v[0])
        pred.append(v[1])
    return pred, true

# def get_seqid(sample_reader,true_reader):
#     s_ids, t_ids = [], []
#     for s_row in sample_reader:
#         s_ids.append(s_row['\ufeffID'])
#     for t_row in true_reader:
#         t_ids.append(t_row['ID'])
#     ids = list(set(s_ids).intersection(set(t_ids)))
#
#     return ids

def get_pred_true(base_dir, pred_path, true_path, sample_path, label_threshold):
    # not all sequences have a ground truth, we should pick up
    pred_file = open(base_dir + pred_path, 'r')
    pred_reader = csv.DictReader(pred_file)
    true_file = open(base_dir + true_path, 'r')
    true_reader = csv.DictReader(true_file)
    sample_file = open(base_dir + sample_path, 'r')
    sample_reader = csv.DictReader(sample_file)
    # seqids = get_seqid(sample_reader,true_reader)
    s_ids, t_ids = [], []
    for s_row in sample_reader:
        s_ids.append(s_row['\ufeffID'])
    for t_row in true_reader:
        t_ids.append(t_row['ID'])
    seqids = list(set(s_ids).intersection(set(t_ids)))
    sample_file.seek(0,0)
    true_file.seek(0,0)
    #
    pred, true = get_value(seqids, pred_reader, true_reader, label_threshold)
    return pred, true

def roc_auc(pred, true):
    # true = np.array([1, 1, 2, 2])
    # pred = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(auc)
    # plot
    plt.xlabel('ground truth')
    plt.ylabel('count')
    plt.scatter(tpr, fpr)
    plt.legend()
    plt.title("ROC curve")
    plt.savefig('/home/data/bedict_reproduce/data/auc.png')
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Train crispr.")
    parser.add_argument('--base_dir', type=str, default="/home/data/bedict_reproduce",
                        help='path to the project.')
    parser.add_argument('--true_path', type=str, default="/data/41467_2021_25375_MOESM2_ESM.csv",
                        help='file path to the truth')
    parser.add_argument('--pred_path', type=str, default="/sample_data/predictions/predictions_predoption_mean.csv",
                        help='file path to the prediction')
    parser.add_argument('--sample_data', type=str, default="/sample_data/abemax_sampledata.csv")
    parser.add_argument('--label_threshold', type=float, default=24.0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    pred, true = get_pred_true(args.base_dir, args.pred_path, args.true_path, args.sample_data, args.label_threshold)
    roc_auc(pred, true)

