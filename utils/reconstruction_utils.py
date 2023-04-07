import numpy as np
from .data_utils import seg_im_class


def vector_reconstruct(vector, samples, k):
    # type: (np.ndarray, np.ndarray, int) -> np.ndarray
    """
    reconstruct the vector by samples: according to the distance between vector
    and samples, we select the top k samples and reconstruct vector all vector there ||vector|| = 1
    Arguments:
        vector: the vector needed reconstructing
        samples:  sample to be selected, include vector
        k: top k
    Returns:
        vector_recon: the reconstructed vector
    """
    num_k = min(k, len(samples) - 1)
    if vector.ndim == 1:
        vector = np.expand_dims(vector, axis=0)
    distance = np.linalg.norm(samples - vector, axis=1)
    index_distance = distance.argsort()
    W = []
    for i in range(1, num_k + 1):
        sample = samples[index_distance[i]]
        w = np.exp(np.sum(sample * vector) / np.sqrt(np.linalg.norm(sample) * np.linalg.norm(vector)) - 1)
        W.append(w)
    W = np.expand_dims(np.array(W), axis=1)
    vector_recon = np.divide(np.sum(samples[index_distance[1:num_k + 1]] * W, axis=0), np.sum(W))
    return vector_recon


def k_near_in_window(data, window_size, top_k):
    # type: (np.ndarray, int, int) -> np.ndarray
    """
    reconstruct the center sample by the top k similar neighbour samples
    in the same window
    Arguments:
        data: the HSI data(M, N, B）
        window_size: the size of window
        top_k: the k most similar neighbour samples
    Returns:
        reconstructed_data: the reconstructed data
    """
    [M, N, B] = data.shape
    reconstructed_data = np.zeros(data.shape)
    for i in range(M):
        for j in range(N):
            vector = data[i, j, :]
            lx = max(0, i - window_size // 2)
            ly = max(0, j - window_size // 2)
            rx = min(M, i + 1 + window_size // 2)
            ry = min(N, j + 1 + window_size // 2)
            samples = data[lx:rx, ly:ry, :].reshape((-1, B))
            vector_recon = vector_reconstruct(vector, samples, top_k)
            reconstructed_data[i, j, :] = vector_recon
    return reconstructed_data


def k_near_in_homogeneity(data, ers_labels, top_k):
    # type: (np.ndarray, np.ndarray, int) -> np.ndarray
    """
    rebuild the center vector by the top k similar neighbours in the homogeneous area
    Arguments:
        data: the HSI data(M, N, B）
        ers_labels: the segment map of HSI(M, N)
        top_k: the k most similar neighbour samples
    Returns:
        reconstructed_data: the reconstructed data
    Returns:
    """
    [M, N, B] = data.shape
    reconstructed_data = np.zeros((M * N, B))
    data_seg = seg_im_class(feature_map=data, label_gt=ers_labels)
    data_seg_index = data_seg['index']
    data_seg_feature = data_seg['feature']
    for key in data_seg_index.keys():
        index = data_seg_index[key]
        samples = data_seg_feature[key]
        for i in range(len(samples)):
            vector = samples[i]
            vector_recon = vector_reconstruct(vector, samples, top_k)
            reconstructed_data[index[i]] = vector_recon
    reconstructed_data = reconstructed_data.reshape((M, N, B))
    return reconstructed_data


def k_near_in_homogeneity_and_window(data, ers_labels, window_size, top_k):
    # type: (np.ndarray, np.ndarray, int, int) -> np.ndarray
    """
    rebuild the center vector by the top k similar neighbours in the homogeneous area
    and window
    Arguments:
        data: the HSI data(M, N, B）
        ers_labels: the segment map of HSI(M, N)
        window_size: the size of window
        top_k: the k most similar neighbour samples
    Returns:
        reconstructed_data: the reconstructed data
    Returns:
    """
    [M, N, B] = data.shape
    reconstructed_data = np.zeros(data.shape)
    for i in range(M):
        for j in range(N):
            vector = data[i, j, :]
            vector_type = ers_labels[i, j]
            lx = max(0, i - window_size // 2)
            ly = max(0, j - window_size // 2)
            rx = min(M, i + 1 + window_size // 2)
            ry = min(N, j + 1 + window_size // 2)
            samples = data[lx:rx, ly:ry, :].reshape((-1, B))
            samples_label = ers_labels[lx:rx, ly:ry].reshape(-1)
            samples = samples[np.where(samples_label == vector_type)]
            vector_recon = vector_reconstruct(vector, samples, top_k)
            reconstructed_data[i, j, :] = vector_recon
    return reconstructed_data
