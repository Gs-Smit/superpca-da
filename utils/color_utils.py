import os
import torch
import cv2
import numpy as np
from sklearn import svm
from utils.data_utils import divide_foreground
from utils.bagging_utils import bagging


def draw_picture(data_name, label, save_path, file_name):
    # type:  (str, np.ndarray, str, str) -> None
    """
    draw the class image according to the labels of all samples and
    store the image(file_name) in save_path
    Arguments:
        data_name: the name of dataset, decide the color of samples in different set
        label: the labels of all samples
        save_path: the path of file for saving image
        file_name: the name of the image, like SuperPCA-DA.jpg
    """
    if not data_name in ['Indian', 'PaviaU']:
        raise ValueError("database does not exist or is not supported")

    color_Indian = [[0, 0, 0], [226, 19, 25], [57, 171, 69], [44, 44, 134], [184, 56, 140], [226, 223, 41],
                    [60, 54, 141], [238, 145, 105], [117, 190, 142], [126, 93, 162], [231, 43, 129],
                    [86, 179, 100], [51, 87, 163], [233, 91, 91], [40, 107, 66], [53, 119, 57],
                    [239, 158, 154]]

    color_PaviaU = [[0, 0, 0], [226, 19, 25], [92, 179, 93], [46, 45, 134], [206, 133, 182],
                    [235, 230, 96], [47, 56, 142], [82, 110, 179], [111, 75, 152],
                    [24, 94, 27]]

    if data_name == 'Indian':
        color_set = color_Indian
    else:
        color_set = color_PaviaU

    label = label.astype(np.int64)
    [h, w] = label.shape
    picture = torch.zeros(3, h, w)

    for i in range(h):
        for j in range(w):
            picture[:, i, j] = torch.Tensor(color_set[label[i, j]])
    picture = picture.permute(1, 2, 0)
    picture = picture.numpy()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cv2.imwrite(os.path.join(save_path, file_name), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))


def color(data, label_gt, model, data_name, save_path, file_name, bag=None):
    # type: (np.ndarray, np.ndarray, object, str, str, str, none) -> None
    """
    Process data and draw images
    Arguments:
        data: the  HSI data
        label_gt: the true label of data, to segment the data into background and foreground
        model: to predict the label of foreground
        data_name: the name of dataset, decide the color of samples in different set
        save_path: the path of file for saving image
        file_name: the name of the image, like SuperPCA-DA.jpg
        bag: bagging model? True/False
    """
    [M, N, _] = data.shape
    if type(model) is svm._classes.SVC:
        foreground = divide_foreground(data, label_gt)
        label_train_pred = model.predict(foreground['data']) + 1
    elif type(model) is list:
        predicts = []
        num_bag = len(model)
        data_bag = data
        for i in range(num_bag):
            data_bag = bag(data_bag)
            foreground = divide_foreground(data_bag, label_gt)
            label_train_pred = model[i].predict(foreground['data']) + 1
            predicts.append(label_train_pred.reshape((-1, 1)))
        predicts = np.concatenate(predicts, axis=1)
        label_train_pred = bagging(predicts)

    label_pred = np.zeros(M * N)
    label_pred[foreground['index']] = label_train_pred
    draw_picture(data_name, label_pred.reshape((M, N)), save_path, file_name)
