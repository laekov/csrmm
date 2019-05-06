from ctypes import cdll
import time
import numpy as np
import sys


lib = cdll.LoadLibrary('./lib{}.so'.format(sys.argv[1]))
lib.prepare()

if __name__ == '__main__':
    times = []
    for i in range(10):
        beg = time.time()
        lib.run()
        end = time.time()
        print('Time = {} s'.format(end - beg))
        times.append(end - beg)

    print('Average time = {} s'.format(np.mean(times)))
