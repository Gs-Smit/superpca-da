from abc import ABC
from sklearn.model_selection import train_test_split

import numpy as np
from numpy.linalg import eig
from collections import OrderedDict

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from utils.data_utils import seg_im_class
from DA.da_base import da_base
from utils.data_utils import samples_divide, feature_norm


def super_pca_parameters_set(database):
    if database == 'Indian':
        num_PC = 30  # THE OPTIMAL PCA DIMENSION.
        num_Pixels = 30  # The value of Sf
    elif database == 'PaviaU':
        num_PC = 5  # THE OPTIMAL PCA DIMENSION.
        num_Pixels = 15  # The value of Sf


def find_k_max_eigen(matrix, num_eigen):
    # type: (np.ndarray, int) -> (np.ndarray, np.ndarray)
    """
    compute the eigenvalues and right eigenvectors of a square array
    and return the top k eigenvalues and right eigenvectors
    Arguments:
        matrix: the N*N size square array
        num_eigen: the num of eigen
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"the shape of the passed matrix does not match. "
                         f"it should be (M, M), while it is {matrix.shape}")
    [_, NN] = matrix.shape
    [S, V] = eig(matrix)
    index = S.argsort()
    eigen_vector = []
    eigen_value = []
    p = NN - 1
    for t in range(num_eigen):
        eigen_vector.append(np.expand_dims(V[:, index[p]], axis=1))
        eigen_value.append(S[index[p]])
        p = p - 1
    eigen_vector = np.concatenate(eigen_vector, axis=1)
    eigen_value = np.array(eigen_value)
    return eigen_vector, eigen_value


def eigenface_f(data_set, num_eigen):
    # type: (np.ndarray, int) -> (np.ndarray, np.ndarray)
    """
    processing raw data and calculating top k eigenvectors and mean sample
    Arguments:
        data_set: the data size is (num_feature, num)
        num_eigen: the num of selected eigen(<=num_feature)
    Returns:
        disc_set: the top k eigenvectors (num_feature, k)
        mean_sample: the mean of data_set(num_feature, )
    """
    if data_set.ndim < 2:
        raise ValueError("if principal component analysis is needed, the data volume shall be greater than 1")
    if data_set.shape[0] < num_eigen:
        raise ValueError(f"the augment num_eigen should less than or equal to the number of feature,"
                         f"while it is {num_eigen}, bigger than {data_set.shape[0]}")
    mean_sample = np.mean(data_set, 1)
    data_set = data_set - mean_sample.reshape((-1, 1))
    R = np.dot(data_set, data_set.T)
    [V, _] = find_k_max_eigen(R, num_eigen)
    disc_set = V
    return disc_set, mean_sample


def noises_gen(num, noise_type='principal_component', noise_range=np.array([0.7, 1.3]), noise_num=3):
    # type: (int, str, np.ndarray, int) -> list
    """
    generate noises by parameter
    Arguments:
        num: the number of noise in noises list
        noise_type: the type of noise 'principal_component' or 'eigenvector'
        noise_range: the range of noise
        noise_num: the number elements of noise effect
    Returns:
        noises: the list of noise
    """
    if noise_type is not 'eigenvector' and noise_type is not 'principal_component':
        raise ValueError(f"noise_type doesn't work, it should be \'principal_component\' or \'eigenvector\',"
                         f"while it is {noise_type}")
    noises = [None]
    noise = OrderedDict()
    noise['noise_type'] = noise_type
    noise['noise_range'] = noise_range
    noise['noise_num'] = noise_num
    for i in range(1, num):
        noises.append(noise)
    return noises


def super_pca_noise(data, labels, num_PC, noise=None):
    # type: (np. ndarray, np.ndarray, int, dir) -> np.ndarray
    """
    add noise into the original HSI data and return the noise-add map
    Arguments:
        data: the (N, M, B) size map, HSI
        labels: the segment label, the sample label means they are in the same homogeneous area
        num_PC: the number of principal component (or eigenvector) to effect
        noise: the information of noise. if noise is None, this function will reconstruction HSI by super_pca
    Returns:
        X: the (N, M, B) size map, added noise
    """
    if noise is not None:
        noise_type = noise['noise_type']
        noise_range = noise['noise_range']
        noise_num = noise['noise_num']
        noise_num = min(noise_num, num_PC)

    [M, N, B] = data.shape
    segment_data = seg_im_class(data, labels)
    segment_index = segment_data['index']
    segment_samples = segment_data['feature']

    X = np.zeros((M * N, B))

    for key in segment_index.keys():
        P, mean_value = eigenface_f(segment_samples[key].T, num_PC)
        PC = np.dot(segment_samples[key], P)
        if noise is not None:
            if noise_type == 'principal_component':
                r = noise_range[0] + (noise_range[1] - noise_range[0]) * np.random.random((PC.shape[0], noise_num))
                PC[:, :noise_num] = PC[:, :noise_num] * r
            elif noise_type == 'eigenvector':
                r = noise_range[0] + (noise_range[1] - noise_range[0]) * np.random.random((P.shape[0], noise_num))
                P[:, :noise_num] = P[:, :noise_num] * r
        x_noise = np.dot(PC, P.T) + mean_value.reshape((1, -1))
        X[segment_index[key], :] = x_noise
    X = feature_norm(X)
    X = X.reshape((M, N, B))
    return X


class superpca_da(da_base):
    def __init__(self, aug_ration, num_percentage, HSI_size, num_PC, rand_pp, ers_labels, noises, trans, low_dim=5,
                 dimension_reduction=False):
        if not aug_ration == len(noises):
            raise ValueError(f"augmentation ratio should equal to the length of noise,"
                             f"while it is {aug_ration} and {len(noises)}")
        self.aug_ratio = aug_ration
        self.num_percentage = num_percentage
        self.HSI_size = HSI_size
        self.num_PC = num_PC
        self.rand_pp = rand_pp
        self.ers_labels = ers_labels
        self.noises = noises
        self.trans = trans
        self.low_dim = low_dim
        self.dimension_reduction = dimension_reduction

    def data_augmentation(self, data, labels):
        if data.ndim == 2:
            B = data.shape[-1]
            data_r = data.reshape((self.HSI_size[0], self.HSI_size[1], B))
        else:
            data_r = data
        data_da = []
        labels_da = []
        for aug in range(self.aug_ratio):
            data_noise = super_pca_noise(data_r, self.ers_labels, self.num_PC, self.noises[aug])

            if self.dimension_reduction:
                shapeor = data_noise.shape
                data_noise = data_noise.reshape(-1, data_noise.shape[-1])
                data_noise = PCA(n_components=self.low_dim).fit_transform(data_noise)
                shapeor = np.array(shapeor)
                shapeor[-1] = self.low_dim
                data_noise = MinMaxScaler().fit_transform(data_noise)
                data_noise = data_noise.reshape(shapeor)

            data_noise = self.trans(data_noise)
            samples = samples_divide(data_noise, labels, self.num_percentage, self.rand_pp)
            data_da.append(samples['data_train'])
            labels_da.append(samples['labels_train'])
        data_da = np.concatenate(data_da, axis=0)
        labels_da = np.concatenate(labels_da, axis=0)
        target = OrderedDict()
        target['data_da'] = data_da
        target['labels_da'] = labels_da
        return target
