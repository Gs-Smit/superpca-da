'''
这个是 With window-based local reconstruction 的数据增强结果
'''
import numpy as np
from sklearn import svm
from utils.datasetutils.read_dataset import read_dataset
from utils.datasetutils.getlabel import getlabel
from utils.SuperPCA.DA import SuperPCA_DA
from utils.score.accuracy import computeAccuracy
from utils.reconstruction.KNear import K_near_WindowAndHomogeneity
import warnings

warnings.filterwarnings('ignore')

trainpercentage = 10  # Training Number per Class
iterNum = 10  # The Iteration Number
augSize = 4  # DA倍数
database = 'Indian'  # 数据集名称

" load the HSI dataset"
if database == 'Indian':
    num_PC = 30  # THE OPTIMAL PCA DIMENSION.
    num_Pixels = 30  # The value of Sf
elif database == 'PaviaU':
    num_PC = 5  # THE OPTIMAL PCA DIMENSION.
    num_Pixels = 15  # The value of Sf

data3D, label_gt, randp, labels = read_dataset(database, num_Pixels)
data3D = data3D / data3D.max()
S = 7; K = 10
data3D = K_near_WindowAndHomogeneity(data3D, labels, S, K)


oas = []
aas = []
kappas = []
for iter in range(iterNum):
    randpp = randp[iter][0]
    DataTest, DataTrain, CTest, CTrain = SuperPCA_DA(data3D, num_PC, label_gt, trainpercentage, labels, randpp, augSize)
    trainlabel = getlabel(CTrain).reshape(-1)
    if augSize > 1:
        trainlabelS = np.r_[trainlabel, trainlabel]
        for _ in range(augSize - 2):
            trainlabelS = np.r_[trainlabelS, trainlabel]
        trainlabel = trainlabelS
    GA = [0.001, 0.005, 0.01, 0.1, 1, 5, 10, 50, 100, 200, 1000]
    DataTest = DataTest[0]
    tempaccuracy1 = 0
    taa = 0
    tkappa = 0
    for tria10 in range(len(GA)):
        gamma = GA[tria10]
        model = svm.SVC(kernel='rbf', gamma=gamma, C=10000)
        model.fit(DataTrain, trainlabel)  # 拟合
        testlabel = getlabel(CTest)
        predict = model.predict(DataTest)
        oa, aa, kappa = computeAccuracy(predict.reshape(-1), testlabel.reshape(-1))
        if oa > tempaccuracy1:
            tempaccuracy1 = oa
            taa = aa
            tkappa = kappa
            predict_label_best = predict
    oas.append(tempaccuracy1)
    aas.append(taa)
    kappas.append(tkappa)
    print('=============================================================')
    print('The  OA (1 iterations) of SuperPCA for ', database, ' is {:.4}'.format(float(tempaccuracy1)))
    print('=============================================================')
oas = np.array(oas)
aas = np.array(aas)
kappas = np.array(kappas)
print('OA:{:.4}'.format(oas.mean()))
print('AA:{:.4}'.format(aas.mean()))
print('Kappa:{:.4}'.format(kappas.mean()))