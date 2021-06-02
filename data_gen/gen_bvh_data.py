import sys

from tqdm import tqdm

from bvh_skeleton import nturgbd_skeleton
from rotation import *

sys.path.extend(['../'])
np.set_printoptions(suppress=True)


def pre_normalization(data):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            skeleton = nturgbd_skeleton.NTURGBDSkeleton()
            channels, header = skeleton.poses2bvh(s[i_s, i_p])
            s[i_s, i_p] = np.asarray(channels)[:, 3:].reshape(T, V, C)

    data = np.transpose(s, [0, 4, 2, 3, 1])  # N, M, T, V, C  to  N, C, T, V, M
    return data


# Estimate skeleton information from 3D pose, converts 3D pose to joint angle and write motion data to bvh file.
if __name__ == '__main__':
    data = np.load('../data/ntu/xsub/val_data_joint.npy')
    pre_normalization(data)
    np.save('../data/ntu/xsub/val_data_bvh.npy', data)

    data = np.load('../data/ntu/xsub/train_data_joint.npy')
    pre_normalization(data)
    np.save('../data/ntu/xsub/train_data_bvh.npy', data)
