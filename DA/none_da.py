from collections import OrderedDict

from DA.da_base import da_base
from utils.data_utils import samples_divide


class none_da(da_base):
    def __init__(self, num_percentage, rand_pp, trans):
        self.num_percentage = num_percentage
        self.rand_pp = rand_pp
        self.trans = trans

    def data_augmentation(self, data, labels):
        data_trans = self.trans(data)
        samples = samples_divide(data_trans, labels, self.num_percentage, self.rand_pp)
        target = OrderedDict()
        target['data_da'] = samples['data_train']
        target['labels_da'] = samples['labels_train']
        return target
