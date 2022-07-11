import numpy as np


def getlabel(classSize):
    for i in range(classSize.shape[0]):
        if i == 0:
            label = i * np.ones((1, int(classSize[i])))
        else:
            label = np.column_stack((label, i * np.ones((1, int(classSize[i])))))
    return label.reshape(-1)
