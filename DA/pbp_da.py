import numpy as np
from collections import OrderedDict

from DA.da_base import da_base
from DA.superpca_da import eigenface_f
from utils.data_utils import samples_divide, feature_norm


def pbp(data, labels):
    num_class = max(labels.reshape(-1)) + 1
    data_new = []
    labels_new = []
    for i in range(num_class):
        data_i = data[labels == i]
        num_i = data_i.shape[0]
        for m in range(num_i):
            for n in range(num_i):
                cub_m = data_i[m]
                cub_n = data_i[n]
                cub = np.concatenate([cub_m, cub_n], axis=1)
                cub = np.expand_dims(cub, axis=0)
                data_new.append(cub)
                labels_new.append(i)
    data_new = np.concatenate(data_new, axis=0)
    labels = np.array(labels)
    return data_new, labels


class pbp_da(da_base):
    def __init__(self, num_percentage, rand_pp, trans):
        self.num_percentage = num_percentage
        self.rand_pp = rand_pp
        self.trans = trans

    def data_augmentation(self, data, labels):
        data_cub = self.trans(data)
        samples = samples_divide(data_cub, labels, self.num_percentage, self.rand_pp)
        data_train = samples['data_train']
        labels_train = samples['labels_train']
        data_train, labels_train = pbp(data_train, labels_train)
        target = OrderedDict()
        target['data_da'] = data_train
        target['labels_da'] = labels_train
        return target
