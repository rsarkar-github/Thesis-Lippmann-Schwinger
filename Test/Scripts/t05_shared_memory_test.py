import numpy as np
from numpy import ndarray
from multiprocessing.managers import SharedMemoryManager


if __name__ == "__main__":

    n = 100
    num_bytes = n * 4

    with SharedMemoryManager() as smm:

        sm = smm.SharedMemory(size=num_bytes)
        data = ndarray(shape=(n,), dtype=np.float32, buffer=sm.buf)
        data *= 0
        data += 1

        for i in range(n):
            data[i] = i

    print(np.sum(data))
