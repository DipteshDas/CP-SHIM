import numpy as np
import os


def create_dir(dir_name):

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def construct_XA(X, A):
    j = A[0]
    XA = np.prod(X[:, j], 1)[:, np.newaxis]
    i = 1
    while i < len(A):
        j = A[i]
        XA = np.column_stack((XA, np.prod(X[:, j], 1)[:, np.newaxis]))
        i += 1
    return XA