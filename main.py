"""
SuperPCA-DA based hyper spectral Imagery Classification(NN)
In this part, we use super pca data augmentation method and 2DNN classifier
to classify HSI samples
"""
import os
import argparse
import warnings
import numpy as np

from utils.data_utils import load_hyper, samples_divide
from utils.reconstruction_utils import k_near_in_homogeneity_and_window, k_near_in_window
from DA.superpca_da import noises_gen, super_pca_noise, superpca_da
from DA.pca_da import pca_noise, pca_da
from DA.sw_da import sw_da
from DA.none_da import none_da
from utils.svm_utils import svm_train, svm_bagging_train
from utils.evaluate_utils import categorized_valuation, overall_valuation_statistics
from utils.color_utils import color, draw_picture
from DA.sw_da import sw

import time
import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from network_files.twoDNN import TwoDNN
from network_files.simpleCNN import SimpleCNN
from utils.nn_utils import image_cube_trans, save_network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='PaviaU', type=str, choices=['Indian', 'PaviaU'], help='Dataset')
    parser.add_argument('--it_num', default=10, type=int, help='the iteration number')
    parser.add_argument('--aug_rat', default=5, type=int, help='the augmentation size')
    parser.add_argument("--visual", action='store_true', help="Visual? Default NO")

    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    data_HSI, labels_gt, rand_p, labels_ers, num_class, num_PC, num_Pixels = load_hyper(args)
    [M, N, _] = data_HSI.shape

    data_HSI = k_near_in_homogeneity_and_window(data_HSI, labels_ers, window_size=7, top_k=10)
    "the experiment of super_pca_da in classifier"
    noises = noises_gen(args.aug_rat, 'principal_component', np.array([0.5, 1.5]), 10)

    valuations = []
    for interr in range(args.it_num):
        "SuperPCA-DA: add noise and get the augmented training data"
        rand_pp = rand_p[interr][0]

        data_gen = superpca_da(args.aug_rat, args.tr_percent, [M, N],
                               num_PC, rand_pp, labels_ers, noises, lambda x: x)
        data_PCR = super_pca_noise(data_HSI, labels_ers, num_PC)

        args.visual = True

        data_labels_da = data_gen.data_augmentation(data_HSI, labels_gt)
        GA = [0.001, 0.005, 0.01, 0.1, 1, 5, 10, 50, 100, 200, 1000]
        data_train_da = data_labels_da['data_da']
        labels_train_da = data_labels_da['labels_da']

        "get the testing data"
        test_samples = samples_divide(data_PCR, labels_gt, args.tr_percent, rand_pp)
        data_test = test_samples['data_test']
        labels_test = test_samples['labels_test']
        "train the svm classifier"
        best_model, best_predict, valuation = svm_train(data_train_da, labels_train_da, data_test,
                                                        labels_test, GA)
        valuations.append(valuation)
        print('=============================================================')
        print(f'The valuation of {args.dataset} in iteration {interr} is\n'
              f'{valuation}')
        print('=============================================================')
        if args.visual and interr == 0:
            color(data_PCR, labels_gt, best_model, args.dataset, f".\\images\\{args.dataset}",
                  f"{args.da_method}_svm.png")

        overall_valuation = overall_valuation_statistics(valuations)
        print('======================CONCLUSION=============================')
        print(f'The overall valuation of {args.dataset} is\n'
              f'{overall_valuation}')
        print('=============================================================')


if __name__ == "__main__":
    main()
