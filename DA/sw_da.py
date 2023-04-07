import numpy as np
from collections import OrderedDict

from DA.da_base import da_base

from utils.reconstruction_utils import k_near_in_window
from utils.data_utils import samples_divide, feature_norm


class sw_da(da_base):
    def __init__(self, aug_ration, num_percentage, window_size, top_k, rand_pp, trans):
        self.aug_ratio = aug_ration
        self.num_percentage = num_percentage
        self.window_Size = window_size
        self.top_k = top_k
        self.rand_pp = rand_pp
        self.trans = trans

    def data_augmentation(self, data, labels):
        data_da = []
        labels_da = []
        data_test = []
        labels_test = []
        data_rc = data
        for aug in range(self.aug_ratio):
            data_rc = k_near_in_window(data_rc, self.window_Size, self.top_k)
            data_rc = feature_norm(data_rc)
            data_rc = self.trans(data_rc)
            samples = samples_divide(data_rc, labels, self.num_percentage, self.rand_pp)
            data_da.append(samples['data_train'])
            labels_da.append(samples['labels_train'])
            data_test.append(samples['data_test'])
            labels_test.append(samples['labels_test'])
        target = OrderedDict()
        target['data_da'] = data_da
        target['labels_da'] = labels_da
        target['data_test'] = data_test
        target['labels_test'] = labels_test
        return target


class sw(object):
    def __init__(self, window_size, top_k):
        self.window_Size = window_size
        self.top_k = top_k

    def __call__(self, data):
        data_rc = k_near_in_window(data, self.window_Size, self.top_k)
        data_rc = feature_norm(data_rc)
        return data_rc
