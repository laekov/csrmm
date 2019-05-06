#!/usr/bin/python3
import pickle
import scipy
import scipy.io as io
import numpy as np


mat = io.loadmat('../catalog/blogcatalog.mat')
def gen_rand_matrix(n, p):
    ptrs = [0]
    indices = []
    for i in range(n):
        idxs = np.nonzero(np.random.uniform(0, 1, n) < p)[0]
        indices += list(idxs)
        ptrs.append(len(indices))
    values = np.ones_like(indices)
    return scipy.sparse.csr_matrix((values, indices, ptrs), shape=(n, n))


opt = 2

if opt == 0:
    adj = gen_rand_matrix(10000, 0.006)
elif opt == 1:
    adj = mat['network']
elif opt == 2:
    with open('bjmap.pkl', 'rb') as f:
        adj = pickle.load(f)

def arr2str(arr):
    return ' '.join([str(x) for x in arr])

if __name__ == '__main__':
    with open('graph.in', 'w') as f:
        f.write('{} {}\n'.format(len(adj.indptr), len(adj.indices)))
        f.write(arr2str(adj.indptr) + '\n')
        f.write(arr2str(adj.indices) + '\n')
        f.write(arr2str(adj.data) + '\n')
