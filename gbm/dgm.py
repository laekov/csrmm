import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv
from preprocess import adj

import numpy as np
import time as time


if __name__ == '__main__':
    g = DGLGraph()
    g.from_scipy_sparse_matrix(adj)
    n = g.number_of_nodes()
    gc = GraphConv(1433, 16, norm=False, bias=False)
    gc.cuda()
    gc.eval()
    h0 = torch.randn(n, 1433).cuda()
    times = []
    for i in range(10):
        t_beg = time.time()
        h1 = gc(h0, g)
        t_end = time.time()
        print('time = {} s'.format(t_end - t_beg))
        times.append(t_end - t_beg)
    print('Average time = {} s'.format(np.mean(times)))

