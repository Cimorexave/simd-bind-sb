import numpy as np
from numba import njit, prange
import time

# create two large arrays of double with the size of 1 million
size = 1_000_000
a = np.random.rand(size).astype(np.float64)
b = np.random.rand(size).astype(np.float64)

@njit(parallel=True, fastmath=True)
def numba_euclidean_distance(a, b):
    n = a.size
    sum = 0.0
    for i in prange(n):
        d = a[i] - b[i]
        sum += d * d
    return np.sqrt(sum)

def np_euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def euclidean_distance(a, b):
    sum = 0.0
    for i in range(a.size):
        d = a[i] - b[i]
        sum += d * d
    return np.sqrt(sum)

if __name__ == "__main__":

    start_time = time.time()
    dist_numba = numba_euclidean_distance(a, b)
    end_time = time.time()
    print(f"Numba distance: {dist_numba:.6f}, Time taken: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    dist_numpy = np_euclidean_distance(a, b)
    end_time = time.time()
    print(f"NumPy distance: {dist_numpy:.6f}, Time taken: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    dist_standard = euclidean_distance(a, b)
    end_time = time.time()
    print(f"Standard distance: {dist_standard:.6f}, Time taken: {end_time - start_time:.4f} seconds")

    # check if all results are approximately equal
    assert np.isclose(dist_numba, dist_numpy), "Numba and NumPy results differ!"
    assert np.isclose(dist_numba, dist_standard), "Numba and standard results differ!"
    print("All results are approximately equal.")