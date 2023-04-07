import numpy as np
from collections import OrderedDict

from DA.da_base import da_base


def mix_up(samples, labels, alpha):
    # type: (np.ndarray, np.ndarray, float) -> (np.ndarray, np.ndarray)
    """
    the data augmentation method:mix up
    Hyper spectral Image Classification With Data Augmentation and Classifier Fusion
    Arguments:
        samples: the samples of HSI
        labels: the labels of samples
        alpha: in [0, 1] argument to control mix up method
    Returns:
        samples_da: the augmented samples (from m original samples to m**2 samples)
        labels_da: the labels of samples_da
    """
    if not len(samples) == len(labels):
        raise ValueError("the number of samples and labels is inconsistent")

    labels = np.array(labels, dtype=np.uint8)
    num_samples = len(samples)
    num_class = np.max(labels) + 1
    labels_one_hot = np.identity(num_class)[labels]
    samples_da = []
    labels_da = []

    for p in range(num_samples):
        for q in range(num_samples):
            new_sample = alpha * samples[p] + (1 - alpha) * samples[q]
            new_label = alpha * labels_one_hot[p] + (1 - alpha) * labels_one_hot[q]
            samples_da.append(np.expand_dims(new_sample, axis=0))
            labels_da.append(np.expand_dims(new_label, axis=0))

    samples_da = np.concatenate(samples_da, axis=0)
    labels_da = np.concatenate(labels_da, axis=0)
    return samples_da, labels_da


class mixup_da(da_base):
    def __init__(self, alpha):
        if alpha > 1 or alpha < 0:
            raise ValueError(f"alpha should in the range [0, 1], while it is {alpha} now")
        self.alpha = alpha

    def data_augmentation(self, data, labels):
        data_da, labels_da = mix_up(data, labels)
        target = OrderedDict()
        target['data_da'] = data_da
        target['labels_da'] = labels_da
        return target
