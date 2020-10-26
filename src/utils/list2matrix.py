import numpy as np
import scipy.sparse as sps


def sparse_column_normalize(sps_mat):
    if sps_mat.format != "csr":
        msg = "Can only column-normalize in place with csr format, not {0}."
        msg = msg.format(sps_mat.format)
        raise ValueError(msg)
    column_norm = np.bincount(sps_mat.indices, weights=sps_mat.data)
    d = sps_mat.diagonal()
    d[np.where(column_norm == 0)] = 1.0
    column_norm[np.where(column_norm == 0)] = 1.0
    lil = sps_mat.tolil()
    lil.setdiag(d)
    sps_mat = lil.tocsr()
    sps_mat.data /= np.take(column_norm, sps_mat.indices)
    return sps_mat


def list2matrix(filename):
    A = np.genfromtxt(
        filename, dtype=[("from", np.intp), ("to", np.intp)], skip_header=4
    )
    m = min(min(A["from"]), min(A["to"]))
    if m > 0:
        A["from"] = A["from"] - 1
        A["to"] = A["to"] - 1
    data = np.ones(len(A["from"]))
    n = max(max(A["from"]), max(A["to"])) + 1
    A = sps.csr_matrix((data, (A["from"], A["to"])), shape=(n, n))
    A = sparse_column_normalize(A)
    return A


if __name__ == "__main__":
    A = list2matrix("./data/Wiki-Vote.txt")
    n = A.shape[1]
    p = 0.15
    y = np.ones(n)
    Ex = lambda x: p * A.dot(x) + (1.0 - p) * np.sum(x)
    # print Ex(y)
    MTx = lambda x: (
        (1.0 - p) * A.T.dot(x) + p * np.sum(x) - x
    )  # An efficient way of implementing M*x
    # print MTx(y)
    csum = A.sum(axis=0)
