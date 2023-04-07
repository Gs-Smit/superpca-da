import numpy as np


def bagging(predicts):
    predict = []
    predicts = np.array(predicts, dtype=np.int)
    for line in predicts:
        predict.append(np.argmax(np.bincount(line)))
    predict = np.array(predict)
    return predict
