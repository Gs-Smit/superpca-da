import numpy as np
from numpy.linalg import eig
from utils.datasetutils.sampledivide import samplesdivide, fea_norm


class Cell:
    def __init__(self, index, Y):
        self.index = index
        self.Y = Y


def seg_im_class(Y, Ln):
    [M, N, B] = Y.shape
    Y_reshape = Y.reshape((M * N, B))
    Gt = Ln.reshape(-1)
    Class = np.unique(Gt)
    Num = Class.shape[0]
    index = []
    Y = []
    for i in range(Num):
        index_i = np.argwhere(Gt == Class[i]).reshape(-1)
        index.append(index_i)
        index_i = list(index_i)
        Y.append(Y_reshape[index_i, :])
    Results = Cell(index, Y)
    return Results


def mysort(S):
    S = list(S)
    S_copy = S.copy()
    S.sort()
    index = []
    for i in S:
        index.append(S_copy.index(i))
    return np.array(S), np.array(index)


def Find_K_Max_Eigen(Matrix, Eigen_NUM):
    [NN, NN] = Matrix.shape
    [S, V] = eig(Matrix)
    [S, index] = mysort(S)
    Eigen_Vector = np.zeros((NN, Eigen_NUM))
    Eigen_Value = np.zeros((1, Eigen_NUM))
    p = NN - 1
    for t in range(Eigen_NUM):
        Eigen_Vector[:, t] = V[:, index[p]]
        Eigen_Value[0, t] = S[p]
        p = p - 1
    return Eigen_Vector, Eigen_Value


def Eigenface_f(Train_SET, Eigen_NUM):
    [NN, Train_NUM] = Train_SET.shape
    Mean_Image = np.mean(Train_SET, 1).reshape((-1, 1))
    Train_SET = Train_SET - Mean_Image
    R = np.dot(Train_SET, Train_SET.T) / (Train_NUM - 1)
    [V, S] = Find_K_Max_Eigen(R, Eigen_NUM)
    disc_value = S
    disc_set = V
    return disc_set


# 数据中心化
def centere_data(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    meanVal = np.tile(meanVal, (rows, 1))
    newdata = dataMat - meanVal
    return newdata, meanVal


def SuperPCA(data, num_PC, labels):
    [M, N, B] = data.shape
    Results_segment = seg_im_class(data, labels)
    Num = len(Results_segment.Y)
    X = np.zeros((M * N, num_PC))
    for i in range(Num):
        P = Eigenface_f(Results_segment.Y[i].T, num_PC)
        PC = np.dot(Results_segment.Y[i], P)
        X[Results_segment.index[i], :] = PC
    X = fea_norm(X)
    PC = X.reshape((M, N, num_PC))
    return PC


def ReBuildPCARand(data, num_PC, labels, seed, model='sample', aug_pc=1):
    [M, N, B] = data.shape
    Results_segment = seg_im_class(data, labels)
    Num = len(Results_segment.Y)
    X = np.zeros((M * N, B))
    for i in range(Num):
        P = Eigenface_f(Results_segment.Y[i].T, num_PC)
        PC = np.dot(Results_segment.Y[i], P)
        np.random.seed(seed * 4 * (i + 1))
        if model == 'sample':
            [Number, _] = PC.shape
            r = 0.9 + 0.2 * np.random.rand(Number, aug_pc)
            PC[:, :aug_pc] = PC[:, :aug_pc] * r[:, :]
        elif model == 'materix':
            [Number, _] = P.shape
            r = 0.9 + 0.2 * np.random.rand(Number, aug_pc)
            P[:, :aug_pc] = P[:, :aug_pc] * r[:, :]
        PC = np.dot(PC, P.T)
        X[Results_segment.index[i], :] = PC
    PC = X.reshape((M, N, B))
    return PC


def ReBuildPCA(data, num_PC, labels):
    [M, N, B] = data.shape
    Results_segment = seg_im_class(data, labels)
    Num = len(Results_segment.Y)
    X = np.zeros((M * N, B))
    for i in range(Num):
        P = Eigenface_f(Results_segment.Y[i].T, num_PC)
        PC = np.dot(Results_segment.Y[i], P)
        PC = np.dot(PC, P.T)
        X[Results_segment.index[i], :] = PC
    PC = X.reshape((M, N, B))
    return PC


