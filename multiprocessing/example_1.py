import multiprocessing as mp
import numpy as np

def square(x): #A
    return np.square(x)

x = np.arange(64) #B
# print(x)

cpu_count = mp.cpu_count()
# print(123)

if __name__ == '__main__':
    pool = mp.Pool(cpu_count) #C
    squared = pool.map(square, [x[8*i:8*i+8] for i in range(6)])
    pool.close()
    pool.join()
    print(squared)