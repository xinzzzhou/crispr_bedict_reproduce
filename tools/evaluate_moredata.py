'''
Evaluate more data
Author: Xin Zhou
Date: 22 Sep, 2021
'''

import argparse
# import numpy as np
import numpy as np
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

def get_pred_true(base_dir, label_threshold, editor, operation):
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
    if operation == "auc":
        return get_value(seqids, pred_reader, true_reader, label_threshold, editor)
    elif operation == "acc":
        return get_position(seqids, pred_reader, true_reader)

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

def get_position(seqids, pred_reader, true_reader):
    # get pred position, begin from 0
    dictlist_pred = defaultdict(list)
    dict_pred = dict()
    dict_pred_position = dict()
    for row in pred_reader:
        if row['id'] in seqids:
            dictlist_pred[row['id']].append(float(row['prob_score_class1']))
            dict_pred[row['id']+"|"+row['prob_score_class1']] = int(row['base_pos'])
    for k, v in dictlist_pred.items():
        max_prob = max(v)
        dict_pred_position[k] = dict_pred.get(k+"|"+ str(max_prob))

    # get true position, begin from 0
    num_zero_seqs = 0
    dict_true_position = dict()
    for row in true_reader:
        if row['ID'] in seqids:
            probs = []
            zero = 0
            for idx in range(1,21):
                prob = float(row['Position_' + str(idx)])
                probs.append(prob)
                if prob == 0.0:
                    zero += 1
            if zero == 20:
                num_zero_seqs += 1
                del dict_pred_position[row['ID']]
            else:
                dict_true_position[row['ID']] = np.argmax(np.array(probs))
    print("num_zeros_seqs : {}".format(num_zero_seqs))
    #         Apos = findA(row['Sequence'], editor)
    #         for idx in Apos:
    #             dictlist_true[row['ID']].append(float(row['Position_' + str(idx)]))
    #             dict_true[row['ID'] + "|" + row['Position_' + str(idx)]] = idx
    # for k, v in dictlist_true.items():
    #     max_prob = max(v)
    #     max_position = dict_true.get(k + "|" + str(max_prob))
    #     print(max_position)
    #     dict_true_position[k] = dict_true.get(k + "|" + str(max_prob))
    #     if not max_prob:
    #         num_zeros += 1
    #         dict_true_position[k] = -1

    positions_pred_true = []
    for k in dict_pred_position.keys():
        l = [dict_pred_position.get(k),dict_true_position.get(k)]
        positions_pred_true.append(l)
    return positions_pred_true


def acc(positions_pred_true):
    nd_positions_pred_true = np.array(positions_pred_true)
    num_right_position = len(np.where(nd_positions_pred_true[:,0] == nd_positions_pred_true[:,1])[0])
    print("accuracy : {:.2f}".format(num_right_position/len(positions_pred_true)))

def parse_args():
    parser = argparse.ArgumentParser(description="Train crispr.")
    parser.add_argument('--base_dir', type=str, default="/home/data/bedict_reproduce",
                        help='path to the project.')
    parser.add_argument('--label_threshold', type=float, default=1.0)
    parser.add_argument('--editor', type=str, default="Target-AID")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # pred, true = get_pred_true(args.base_dir, args.label_threshold, args.editor, "auc")
    # roc_auc(pred, true, args.base_dir, args.editor)
    positions_pred_true = get_pred_true(args.base_dir, args.label_threshold, args.editor, "acc")
    acc(positions_pred_true)
