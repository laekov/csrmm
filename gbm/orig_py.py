import torch
import torch.nn as nn
from preprocess import adj

import time
import numpy as np


adj = adj.tocoo()


class GCN(nn.Module):
    def __init__(self, in_fea, ou_fea, adj):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_fea, ou_fea, bias=None)
        self.fc.cuda()
        self.g = torch.sparse.FloatTensor(
                torch.LongTensor([adj.col, adj.row]),
                torch.FloatTensor(adj.data),
                adj.shape).cuda()
        # self.g.cuda()

    def forward(self, h):
        h = self.fc(h)
        h = torch.sparse.mm(self.g, h)
        return h


if __name__ == '__main__':
    n = adj.shape[0]
    gcn = GCN(1433, 16, adj)
    gcn.cuda()
    gcn.eval()
    h0 = torch.randn(n, 1433).cuda()
    times = []
    for i in range(10):
        t_beg = time.time()
        h1 = gcn(h0)
        t_end = time.time()
        print('time = {} s'.format(t_end - t_beg))
        times.append(t_end - t_beg)
    print('Average time = {} s'.format(np.mean(times)))

