from sklearn.metrics import confusion_matrix
import numpy as np


def computeAccuracy(y_pred, Y):
    '''

    :param y_pred: 预测标签值
    :param Y: 真实值
    :return: 返回p0, AA, Kappa 值
    '''
    count = 0
    for i in range(Y.shape[0]):
        if (Y[i] == y_pred[i]):
            count += 1

    p0 = count * 1.0 / Y.shape[0]

    confusion = confusion_matrix(Y, y_pred)
    sumkind = np.sum(confusion, axis=1)
    aa = 0
    kinds = sumkind.shape[0]
    for i in range(kinds):
        aa += confusion[i][i] / sumkind[i]
    aa /= kinds

    pe = 0
    for i in range(kinds):
        pe += sumkind[i] * confusion[i][i]
    pe /= sum(sumkind) * sum(sumkind)
    Kappa = (p0 - pe) / (1 - pe)

    return p0, aa, Kappa
