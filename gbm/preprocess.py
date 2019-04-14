import scipy.io as io


mat = io.loadmat('../catalog/blogcatalog.mat')
adj = mat['network']


def arr2str(arr):
    return ' '.join([str(x) for x in arr])

if __name__ == '__main__':
    with open('graph.in', 'w') as f:
        f.write('{} {}\n'.format(len(adj.indptr), len(adj.indices)))
        f.write(arr2str(adj.indptr) + '\n')
        f.write(arr2str(adj.indices) + '\n')
        f.write(arr2str(adj.data) + '\n')
