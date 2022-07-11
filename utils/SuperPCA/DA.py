import numpy as np
from utils.datasetutils.sampledivide import samplesdivide
from utils.SuperPCA.superPCA import ReBuildPCA, ReBuildPCARand


def SuperPCA_DA(data3D, num_PC, label_gt, trainpercentage, labels, randpp, augSize):
    DataTest = []
    for seed in range(augSize):
        if seed == 0:
            dataDR = ReBuildPCA(data3D, num_PC, labels)
            [datatest, DataTrain, CTest, CTrain] = samplesdivide(dataDR, label_gt, trainpercentage, randpp)
            DataTest.append(datatest)
        else:
            dataDR = ReBuildPCARand(data3D, num_PC, labels, seed * 10)
            [datatest, DataTrainTemp, _, _] = samplesdivide(dataDR, label_gt, trainpercentage, randpp)
            DataTrain = np.row_stack((DataTrain, DataTrainTemp))
            DataTest.append(datatest)
    return DataTest, DataTrain, CTest, CTrain
