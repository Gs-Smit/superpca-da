import numpy as np


def fea_norm(fea):
    [nSamp, nFea] = fea.shape
    fea_co = np.zeros((nSamp, nFea))
    for i in range(nSamp):
        fea_co[i, :] = fea[i, :] / max(1e-12, np.linalg.norm(fea[i, :]))
    return fea_co


def choiceKind(randpp, num_kind):
    index = []
    num = len(randpp)
    ch = np.zeros((1, num))
    for i in range(num):
        index.append(randpp[i].shape[1])
    ind = index.copy()
    index.sort(reverse=True)
    for i in range(num_kind):
        ch[0, ind.index(index[i])] = 1
    return ch.reshape(-1)


def samplesdivide(indian_pines_corrected, indian_pines_gt, train, randpp):
    [m, n, p] = indian_pines_corrected.shape
    CTrain = []
    CTest = []
    data_col = indian_pines_corrected.reshape((m * n, p))
    indian_pines_gt = indian_pines_gt.reshape(-1)
    num = 0
    for i in range(1, int(max(indian_pines_gt)) + 1):
        ci = np.argwhere(indian_pines_gt == i).shape[0]
        v = np.argwhere(indian_pines_gt == i)
        datai = data_col[list(v.reshape(-1)), :]
        if train > 1:
            cTrain = round(train)
        elif train < 1:
            cTrain = round(ci * train)
        if train > ci / 2:
            cTrain = round(ci / 2)
        cTest = ci - cTrain
        CTrain.append(cTrain)
        CTest.append(cTest)
        index = randpp[i - 1].reshape(-1) - 1
        if num == 0:
            DataTest = datai[index[:cTest], :]
        else:
            DataTest = np.row_stack((DataTest, datai[index[:cTest], :]))
        if num == 0:
            DataTrain = datai[index[cTest:cTest + cTrain], :]
        else:
            DataTrain = np.row_stack((DataTrain, datai[index[cTest:cTest + cTrain], :]))
        num = num + 1
    DataTest = fea_norm(DataTest)
    DataTrain = fea_norm(DataTrain)
    CTest = np.array(CTest)
    CTrain = np.array(CTrain)
    return DataTest, DataTrain, CTest, CTrain
