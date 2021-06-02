import pickle

import numpy as np
from tqdm import tqdm

label = open('./data/ntu/xsub/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('./work_dir/ntu_ShiftGCN_bone_xsub/eval_results/best_acc.pkl', 'rb')  # _vit_scaled
r1 = list(pickle.load(r1).items())
r2 = open('./work_dir/ntu_ShiftGCN_vit_xsub/eval_results/best_acc.pkl', 'rb')
r2 = list(pickle.load(r2).items())

alpha = [0.5, 0.5, 0.5, 0.5]  # same hyperparameter is used for NTU/NTU120/NW-UCLA

right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    r = r11 * alpha[0] + r22 * alpha[1]
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print('top1: ', acc)
print('top5: ', acc5)
