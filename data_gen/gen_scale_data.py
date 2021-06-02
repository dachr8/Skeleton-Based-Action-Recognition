import random
import sys

sys.path.extend(['../'])
from rotation import *
from tqdm import tqdm


def pre_normalization(data):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print(np.median(s), np.mean(s), np.max(s))
    print('scale')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            s[i_s, i_p] *= random.uniform(0, 2)
    print(np.median(s), np.mean(s), np.max(s))
    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data


if __name__ == '__main__':
    data = np.load('../data/ntu/xsub/val_data_joint.npy')
    pre_normalization(data)
    np.save('../data/ntu/xsub/val_data_scale.npy', data)

    data = np.load('../data/ntu/xsub/train_data_joint.npy')
    pre_normalization(data)
    np.save('../data/ntu/xsub/train_data_scale.npy', data)
