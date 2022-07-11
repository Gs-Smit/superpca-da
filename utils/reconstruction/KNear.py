import numpy as np

def mysort(S):
    S_copy = S.copy()
    S.sort()
    index = []
    for i in S:
        index.append(S_copy.index(i))
    return S, index

def fea_norm(fea):
    [nSamp, nFea] = fea.shape
    fea_co = np.zeros((nSamp, nFea))
    for i in range(nSamp):
        fea_co[i, :] = fea[i, :] / max(1e-12, np.linalg.norm(fea[i, :]))
    return fea_co


def GetHomogeneousArea(labels):
    [M, N] = labels.shape
    Dictionary = {}
    for i in range(M):
        for j in range(N):
            c = labels[i][j]
            if Dictionary.get(c) == None:
                Dictionary[c] = [[i, j]]
            else:
                Dictionary[c].append([i, j])
    return Dictionary


def K_near_WindowAndHomogeneity(data, labels, S, K):
    [M, N, B] = data.shape
    data = data.reshape((M * N, B))
    data = fea_norm(data)
    data = data.reshape((M, N, B))
    X = np.zeros((M, N, B))
    for i in range(M):
        for j in range(N):
            pixel1 = data[i, j, :]
            Behind = []
            Distance = []
            for l in range(-S // 2 + 1, S // 2 + 1):
                for m in range(-S // 2 + 1, S // 2 + 1):
                    if i + l < 0 or j + m < 0 \
                            or i + l >= M or j + m >= N \
                            or (l == 0 and m == 0):
                        continue
                    elif labels[i + l][j + m] != labels[i][j]:
                        continue
                    else:
                        pixel2 = data[i + l, j + m, :]
                        distance = np.linalg.norm(pixel1 - pixel2)
                        Behind.append([l, m])
                        Distance.append(distance)
            W = []
            pixel = np.zeros((1, 1, B))
            if len(Distance) > K:
                [_, myIndex] = mysort(Distance)
                for myi in range(K):
                    indexx = myIndex[myi]
                    pixel2 = data[i + Behind[indexx][0], j + Behind[indexx][1], :]
                    w = np.exp(np.sum(pixel1 * pixel2) / np.sqrt(np.linalg.norm(pixel1) * np.linalg.norm(pixel2)) - 1)
                    pixel = pixel + w * pixel2.reshape((1, 1, B))
                    W.append(w)
                pixel = pixel / sum(W)
            else:
                for index in range(len(Distance)):
                    pixel2 = data[i + Behind[index][0], j + Behind[index][1], :]
                    w = np.exp(np.sum(pixel1 * pixel2) / np.sqrt(np.linalg.norm(pixel1) * np.linalg.norm(pixel2)) - 1)
                    pixel = pixel + w * pixel2.reshape((1, 1, B))
                    W.append(w)
                pixel = pixel / sum(W)
            X[i, j, :] = pixel
    X = X.reshape((M * N, B))
    X = fea_norm(X)
    X = X.reshape((M, N, B))
    return X


def K_near_Window(data, S, K):
    [M, N, B] = data.shape
    data = data.reshape((M * N, B))
    data = fea_norm(data)
    data = data.reshape((M, N, B))
    X = np.zeros((M, N, B))
    for i in range(M):
        for j in range(N):
            pixel1 = data[i, j, :]
            Behind = []
            Distance = []
            for l in range(-S // 2 + 1, S // 2 + 1):
                for m in range(-S // 2 + 1, S // 2 + 1):
                    if i + l < 0 or j + m < 0 \
                            or i + l >= M or j + m >= N \
                            or (l == 0 and m == 0):
                        continue
                    else:
                        pixel2 = data[i + l, j + m, :]
                        distance = np.linalg.norm(pixel1 - pixel2)
                        Behind.append([l, m])
                        Distance.append(distance)
            W = []
            pixel = np.zeros((1, 1, B))
            if len(Distance) > K:
                [_, myIndex] = mysort(Distance)
                for myi in range(K):
                    indexx = myIndex[myi]
                    pixel2 = data[i + Behind[indexx][0], j + Behind[indexx][1], :]
                    w = np.exp(np.sum(pixel1 * pixel2) / np.sqrt(np.linalg.norm(pixel1) * np.linalg.norm(pixel2)) - 1)
                    pixel = pixel + w * pixel2.reshape((1, 1, B))
                    W.append(w)
                pixel = pixel / sum(W)
            else:
                for index in range(len(Distance)):
                    pixel2 = data[i + Behind[index][0], j + Behind[index][1], :]
                    w = np.exp(np.sum(pixel1 * pixel2) / np.sqrt(np.linalg.norm(pixel1) * np.linalg.norm(pixel2)) - 1)
                    pixel = pixel + w * pixel2.reshape((1, 1, B))
                    W.append(w)
                pixel = pixel / sum(W)
            X[i, j, :] = pixel
    X = X.reshape((M * N, B))
    X = fea_norm(X)
    X = X.reshape((M, N, B))
    return X
