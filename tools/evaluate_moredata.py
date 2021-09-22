'''
Evaluate more data
Author: Xin Zhou
Date: 22 Sep, 2021
'''

import argparse
# import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import os

def findA(seq, editor):
    idx = 0
    position = []
    for s in seq:
        idx += 1
        #
        if editor in {'ABEmax', 'ABE8e'}:
            target_base = 'A'
        elif editor in {'CBE4max', 'Target-AID'}:
            target_base = 'C'
        #
        if s == target_base:
            position.append(idx)
    return position

def get_value(seqids, pred_reader, true_reader, label_threshold, editor):
    dict_true_pred = defaultdict(list)

    for row in true_reader:
        if row['ID'] in seqids:
            Apos = findA(row['Sequence'], editor)
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

def get_pred_true(base_dir, label_threshold, editor):
    # not all sequences have a ground truth, we should pick up
    pred_file = open(os.path.join(base_dir, 'data/test_data', editor, 'predictions/predictions_predoption_mean.csv'), 'r')
    pred_reader = csv.DictReader(pred_file)
    true_file = open(os.path.join(base_dir, 'data/test_data', editor, 'perbase.csv'), 'r')
    true_reader = csv.DictReader(true_file)
    sample_file = open(os.path.join(base_dir, 'data/test_data', editor, 'perbase_testdata_format.csv'), 'r')
    sample_reader = csv.DictReader(sample_file)
    # seqids = get_seqid(sample_reader,true_reader)
    s_ids, t_ids = [], []
    for s_row in sample_reader:
        s_ids.append(s_row['ID'])
    for t_row in true_reader:
        t_ids.append(t_row['ID'])
    seqids = list(set(s_ids).intersection(set(t_ids)))
    sample_file.seek(0,0)
    true_file.seek(0,0)
    #
    pred, true = get_value(seqids, pred_reader, true_reader, label_threshold, editor)
    return pred, true

def roc_auc(pred, true, base_dir, editor):
    fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(auc)
    # plot
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.scatter(fpr, tpr)
    plt.legend()
    plt.title("ROC curve")
    plt.savefig(os.path.join(base_dir,'data/test_data', editor, 'roc_'+str(auc)+'.png'))
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Train crispr.")
    parser.add_argument('--base_dir', type=str, default="/home/data/bedict_reproduce",
                        help='path to the project.')
    parser.add_argument('--label_threshold', type=float, default=1.0)
    parser.add_argument('--editor', type=str, default="Target-AID")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    pred, true = get_pred_true(args.base_dir, args.label_threshold, args.editor)
    roc_auc(pred, true, args.base_dir, args.editor)
