import numpy as np

GaussianFilter256 = (1 / 256) * np.asarray([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], np.float32)

GaussianFilter159 = (1 / 159) * np.asarray([
    [2, 4, 5, 4, 2],
    [4, 9, 12, 9, 2],
    [5, 12, 15, 12, 5],
    [4, 9, 12, 9, 4],
    [2, 4, 5, 4, 2]
], np.float32)

GaussianFilter = np.asarray([
    [0.0947416, 0.118318, 0.0947416],
    [0.118318, 0.147761, 0.118318],
    [0.0947416, 0.118318, 0.0947416]
])

SobelXFilter = np.asarray([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], np.float32)

SobelYFilter = np.asarray([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
], np.float32)

PrewittXFilter = np.asarray([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], np.float32)

PrewittYFilter = np.asarray([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])
