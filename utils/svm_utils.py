import numpy as np
from sklearn import svm
from utils.evaluate_utils import categorized_valuation
from utils.bagging_utils import bagging


def svm_train(data_train, labels_train, data_test, labels_test, GA):
    """
    search the best gamma for the svm and return the param
    Arguments:
        data_train: data for train
        labels_train: labels for train
        data_test: data for test
        labels_test: labels for test
        GA: list of gammas
    Returns:
        best_model: the best svm models
        best_valuation: the best valuation result
    """
    best_valuation = None
    best_model = None
    best_predict = None
    for tria10 in range(len(GA)):
        gamma = GA[tria10]
        model = svm.SVC(kernel='rbf', gamma=gamma, C=10000)
        model.fit(data_train, labels_train)  # 拟合
        predict = model.predict(data_test)
        valuation = categorized_valuation(y_true=labels_test, y_pred=predict)
        if best_valuation is None or valuation['oa'] > best_valuation['oa']:
            best_valuation = valuation
            best_model = model
            best_predict = predict
    return best_model, best_predict, best_valuation


def svm_bagging_train(data_train, labels_train, data_test, labels_test, GA):
    num = len(data_train)
    predicts = []
    models = []
    for i in range(num):
        best_model, best_predict, best_valuation = svm_train(data_train[i], labels_train[i], data_test[i],
                                                             labels_test[i], GA)
        predicts.append(best_predict.reshape((-1, 1)))
        models.append(best_model)
    predicts = np.concatenate(predicts, axis=1)
    predict = bagging(predicts)
    valuation = categorized_valuation(y_true=labels_test[0], y_pred=predict)
    return models, predict, valuation
