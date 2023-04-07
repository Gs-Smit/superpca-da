from collections import OrderedDict
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, precision_recall_fscore_support


def categorized_valuation(y_true, y_pred):
    # type: (np.ndarray, np.ndarray) -> dir[str, float]
    """
    valuate the classification result by accuracy, average recall and kappa
    Arguments:
        y_true: the true labels
        y_pred: the predicted labels
    Return:
        valuation: the dict of p0, aa and kappa
    """
    oa = accuracy_score(y_true, y_pred)
    aa = recall_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    labels = [i for i in range(int(y_true.max()) + 1)]
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=labels, average=None
    )
    valuation = OrderedDict()
    valuation['oa'] = oa
    valuation['aa'] = aa
    valuation['kappa'] = kappa
    valuation['p_class'] = p_class
    valuation['r_class'] = r_class
    valuation['f_class'] = f_class
    valuation['support_micro'] = support_micro
    return valuation


def overall_valuation_statistics(valuations):
    # type: (list[dict]) -> dict
    """
    Overall accuracy of statistics
    Arguments:
        valuations: a list of valuations
    Returns:
        overall_valuation: the result of overall accuracy
    """
    oa = 0
    aa = 0
    kappa = 0
    p_class = 0
    for val in valuations:
        oa += val['oa']
        aa += val['aa']
        kappa += val['kappa']
        p_class += val['p_class']
    oa /= len(valuations)
    aa /= len(valuations)
    kappa /= len(valuations)
    p_class /= len(valuations)
    overall_valuation = OrderedDict()
    overall_valuation['oa'] = oa
    overall_valuation['aa'] = aa
    overall_valuation['kappa'] = kappa
    overall_valuation['p_class'] = p_class
    return overall_valuation
