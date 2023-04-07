import os
import scipy.io as scio
import numpy as np
from collections import OrderedDict
from utils.parameter_utils import super_pca_parameters


def load_data(database, Sf=10):
    # type:  (str, int) -> dict[str, np.ndarray]
    """
    Read the dataset of Indian Pines and PaviaU and load them into the OrderedDict
    [str, np.ndarray] target.
    Arguments:
        database (str): the name of the dataset
        Sf(int): the number of the homogeneous areas in HSI
    Returns:
        data (OrderedDict[str, np.ndarray]): the whole data dict for the dataset
    """
    if not database in ['Indian', 'PaviaU']:
        raise ValueError("database does not exist or is not supported")

    cur_path = os.path.abspath(os.path.dirname(__file__))
    data_path = cur_path[:cur_path.find("SuperPCA-DA\\") + len("SuperPCA-DA\\")]
    data_path = os.path.join(data_path, f'dataset\\{database}\\')

    if database == 'Indian':
        data = scio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels_gt = scio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        rand_p = scio.loadmat(os.path.join(data_path, 'Indian_pines_randp.mat'))['randp'][0]
        labels_ers = scio.loadmat(os.path.join(data_path, f'ERSlabel/Indian_labels_num_Pixels_{Sf}.mat'))[
            'labels']
    elif database == 'PaviaU':
        data = scio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels_gt = scio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        rand_p = scio.loadmat(os.path.join(data_path, 'PaviaU_randp.mat'))['randp'][0]
        labels_ers = scio.loadmat(os.path.join(data_path, f'ERSlabel/PaviaU_labels_num_Pixels_{Sf}.mat'))[
            'labels']
    else:
        print("NO DATASET")
        exit()
    num_class = len(np.unique(labels_gt)) - 1
    return data, labels_gt, rand_p, labels_ers, num_class


def load_hyper(args, normal=True):
    """parameter setting"""
    num_PC, num_Pixels = super_pca_parameters(args.dataset)
    " load the HSI dataset"
    data_HSI, labels_gt, rand_p, labels_ers, num_class = load_data(args.dataset, num_Pixels)
    " the process of K neighbor reconstruction"
    [M, N, B] = data_HSI.shape
    if normal:
        data_HSI = data_HSI / data_HSI.max()
        data_HSI = feature_norm(data_HSI.reshape((-1, B))).reshape((M, N, B))
    return data_HSI, labels_gt, rand_p, labels_ers, num_class, num_PC, num_Pixels


def get_label(class_size):
    # type: (np.ndarray) -> np.ndarray
    """
    Set the label of samples into out(np.ndarray)
    Arguments:
        class_size:records the number of samples in each category
    Returns:
        out:records the class of each sample
    """
    num_class = len(class_size)
    result = []
    for i in range(num_class):
        result.append(i * np.ones(class_size[i]))
    out = np.concatenate(result, axis=0)
    return out


def feature_norm(feature):
    # type: (np.ndarray) -> np.ndarray
    """
    Normalize the features of samples
    Arguments:
        feature: N*M matrix record N samples of M features
    Returns:
        fea_co: the normalized samples N*M
    """
    nshape = feature.shape
    if feature.ndim == 3:
        feature = feature.reshape((-1, nshape[-1]))
    fea_co = np.linalg.norm(feature, axis=1)
    fea_co = np.where(fea_co < 1e-6, 1e-6, fea_co)
    fea_co = feature / fea_co.reshape((-1, 1))
    fea_co = fea_co.reshape(nshape)
    return fea_co


def seg_im_class(feature_map, label_gt):
    # type: (np.ndarray, np.ndarray) -> dict[str, dict[str, np.ndarray]]
    """
    segment the HSI into different class by the ground label and store the result
    into an OrderedDict
    Arguments:
        feature_map: the features of the HSI(size:M*N*B)
        label_gt: labels corresponding to HSI feature_map
    Returns:
        result: the OrderDict, include the index of samples in different class
        result = {
        'index':{'0': np.ndarray, '1':...}
        'feature':{'0': np.ndarray, '1':...}
        }
    """
    [M, N, B] = feature_map.shape
    feature_map_reshape = feature_map.reshape((M * N, B))
    Gt = label_gt.reshape(-1)
    Class = np.unique(Gt)
    num_class = Class.shape[0]
    result = OrderedDict()
    index = {}
    feature = {}
    for i in range(num_class):
        index_i = np.argwhere(Gt == Class[i]).reshape(-1)
        feature_i = feature_map_reshape[index_i]
        if len(index_i) == 1:
            feature_i = np.expand_dims(feature_i, axis=0)
        index[f'{i}'] = index_i
        feature[f'{i}'] = feature_i
    result['index'] = index
    result['feature'] = feature
    return result


def samples_divide(feature_map, label_gt, num_train, rand_pp):
    # type: (np.ndarray, np.ndarray, int, np.ndarray) -> dict[str, np.ndarray]
    """
    Divide the samples into train and test according to the number of training samples
    and the sequence of randpp
    Arguments:
        feature_map: the features map of the HSI
        label_gt: the true label map of HSI
        num_train: the number of training samples
        rand_pp: the rand sequence of samples in different class
    Returns:
        train_test_samples (OrderedDict[str, np.ndarray]): the whole data dict for the training samples
        and testing samples
    """
    if feature_map.ndim == 3:
        [m, n, p] = feature_map.shape
        data_col = feature_map.reshape((m * n, p))
    else:
        data_col = feature_map
    label_gt = label_gt.reshape(-1)
    data_train = []
    data_test = []
    c_train = []
    c_test = []
    for i in range(1, int(max(label_gt)) + 1):
        v = np.argwhere(label_gt == i).reshape(-1)
        ci = len(v)
        if num_train > 1:
            cTrain = round(num_train)
        elif num_train < 1:
            cTrain = round(ci * num_train)
        if num_train > ci / 2:
            cTrain = round(ci / 2)
        cTest = ci - cTrain
        c_train.append(cTrain)
        c_test.append(cTest)
        index = rand_pp[i - 1].reshape(-1) - 1
        data_train.append(v[index[cTest:cTest + cTrain]])
        data_test.append(v[index[:cTest]])
    data_train = data_col[np.concatenate(data_train, axis=0)]
    label_train = get_label(np.array(c_train))
    data_test = data_col[np.concatenate(data_test, axis=0)]
    label_test = get_label(np.array(c_test))

    train_test_samples = OrderedDict()
    train_test_samples['data_train'] = data_train
    train_test_samples['labels_train'] = label_train
    train_test_samples['data_test'] = data_test
    train_test_samples['labels_test'] = label_test
    return train_test_samples


def divide_foreground(data, label_gt):
    # type: (np.ndarray, np.ndarray) -> dir[str, np.ndarray]
    """
    divide foreground part of the HSI
    Arguments:
         data: the HSI map (M, N, B)
         label_gt: the true label of data
    Returns:
        foreground: the foreground information
    """
    if data.ndim == 3:
        [M, N, B] = data.shape
        data_r = data.reshape((M * N, B))
    else:
        data_r = data
    label_gt_r = label_gt.reshape(-1)
    index_foreground = np.argwhere(label_gt_r > 0).reshape(-1)
    data_foreground = data_r[index_foreground]
    foreground = OrderedDict()
    foreground['data'] = data_foreground
    foreground['index'] = index_foreground
    return foreground
