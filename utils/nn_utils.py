import os
import torch
import numpy as np


def pad_with_zeros(X, margin=2):
    """
    paddle the original data with zero
    Argument:
        X: the original data
        margin:
    Returns:
        new_X : the paddled X
    """
    [M, N, B] = X.shape
    new_X = np.zeros((M + 2 * margin, N + 2 * margin, B))
    new_X[margin:M + margin, margin:N + margin, :] = X
    return new_X


def create_image_cubes(X, window_size=5):
    # type: (np.ndarray, int) -> (np.ndarray)
    """
    get the images cubes of the HSI
    Arguments:
        X: the HSI data
        window_size: the size of window
    Returns:
        patches_data: the list of patches data
    """
    [M, N, B] = X.shape
    margin = int((window_size - 1) / 2)
    zero_padded_X = pad_with_zeros(X, margin=margin)
    # split patches
    patches_data = np.zeros((M * N, window_size, window_size, B))
    patch_index = 0
    for r in range(margin, M + margin):
        for c in range(margin, N + margin):
            patch = zero_padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patches_data[patch_index, :, :, :] = patch
            patch_index = patch_index + 1
    return patches_data


class image_cube_trans:
    def __init__(self, window_size=5):
        self.window_size = window_size

    def __call__(self, X):
        patches_data = create_image_cubes(X, self.window_size)
        patches_data = np.transpose(patches_data, (0, 3, 1, 2))
        return patches_data


def save_network(model, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, name))
