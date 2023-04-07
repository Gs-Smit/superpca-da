import numpy as np
from collections import OrderedDict

from DA.da_base import da_base
from DA.superpca_da import eigenface_f
from utils.data_utils import samples_divide, feature_norm


def pca_noise(data, num_PC, noise=None):
    if noise is not None:
        noise_type = noise['noise_type']
        noise_range = noise['noise_range']
        noise_num = noise['noise_num']
        noise_num = min(noise_num, num_PC)

    [M, N, B] = data.shape

    data_row = data.reshape((-1, B))

    P, mean_value = eigenface_f(data_row.T, num_PC)
    PC = np.dot(data_row, P)

    if noise is not None:
        if noise_type == 'principal_component':
            r = noise_range[0] + (noise_range[1] - noise_range[0]) * np.random.random((PC.shape[0], noise_num))
            PC[:, :noise_num] = PC[:, :noise_num] * r
        elif noise_type == 'eigenvector':
            r = noise_range[0] + (noise_range[1] - noise_range[0]) * np.random.random((P.shape[0], noise_num))
            P[:, :noise_num] = P[:, :noise_num] * r

    x_noise = np.dot(PC, P.T) + mean_value.reshape((1, -1))
    x_noise = feature_norm(x_noise)
    x_noise = x_noise.reshape((M, N, B))
    return x_noise


class pca_da(da_base):
    def __init__(self, aug_ration, num_percentage, num_PC, rand_pp, noises, trans):
        if not aug_ration == len(noises):
            raise ValueError(f"augmentation ratio should equal to the length of noise,"
                             f"while it is {aug_ration} and {len(noises)}")
        self.aug_ratio = aug_ration
        self.num_percentage = num_percentage
        self.num_PC = num_PC
        self.rand_pp = rand_pp
        self.noises = noises
        self.trans = trans

    def data_augmentation(self, data, labels):
        data_da = []
        labels_da = []
        for aug in range(self.aug_ratio):
            data_noise = pca_noise(data, self.num_PC, self.noises[aug])
            data_noise = self.trans(data_noise)
            samples = samples_divide(data_noise, labels, self.num_percentage, self.rand_pp)
            data_da.append(samples['data_train'])
            labels_da.append(samples['labels_train'])
        data_da = np.concatenate(data_da, axis=0)
        labels_da = np.concatenate(labels_da, axis=0)
        target = OrderedDict()
        target['data_da'] = data_da
        target['labels_da'] = labels_da
        return target
