import scipy.io as scio
import numpy as np

def read_dataset(database, Sf):
    if database == 'Indian':
        _data3D = scio.loadmat('../dataset/Indian/Indian_pines_corrected.mat')['indian_pines_corrected']
        _label_gt = scio.loadmat('../dataset/Indian/Indian_pines_gt.mat')['indian_pines_gt']
        _randp = scio.loadmat('../dataset/Indian/Indian_pines_randp.mat')['randp'][0]
        if Sf != 1:
            _labels = scio.loadmat('../dataset/Indian/ERSlabel/Indian_labels_num_Pixels_' + str(Sf) + '.mat')['labels']
        else:
            _labels = np.zeros(_label_gt.shape)
    elif database == 'PaviaU':
        _data3D = scio.loadmat('../dataset/PaviaU/PaviaU.mat')['paviaU']
        _label_gt = scio.loadmat('../dataset/PaviaU/PaviaU_gt.mat')['paviaU_gt']
        _randp = scio.loadmat('../dataset/PaviaU/PaviaU_randp.mat')['randp'][0]
        _labels = scio.loadmat('../dataset/PaviaU/ERSlabel/PaviaU_labels_num_Pixels_' + str(Sf) + '.mat')['labels']
    return _data3D, _label_gt, _randp, _labels
