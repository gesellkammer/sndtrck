#cython: embedsignature=True
#cython: infer_types=True
#cython: c_string_type=str, c_string_encoding=ascii
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython
from numpy.math cimport INFINITY
import array

def test(int a, int b):
    return a + b

def _frametimes_iter(int N, np.ndarray[double, ndim=1] sorted_t, np.ndarray[long, ndim=1] sorted_idx):
    cdef long i, idx
    cdef double t
    cdef list frametimes = [sorted_t[0]]
    cdef np.ndarray[np.int8_t, ndim=1] present = np.zeros((N,), dtype=np.int8)
    assert sorted_t.size == sorted_idx.size
    for i in range(sorted_t.shape[0]):
        idx = sorted_idx[i]
        if present[idx] == 1:
            frametimes.append(sorted_t[i])
            present.fill(0)
        else:
            present[idx] = 1
    return frametimes


def minmax1d(double[:] a not None):
    cdef int size = a.shape[0]
    cdef size_t i
    cdef double x0 = INFINITY
    cdef double x1 = -INFINITY
    cdef double x
    for i in range(size):
        x = a[i]
        if x < x0:
            x0 = x
        elif x > x1:
            x1 = x
    return x0, x1

def fillrow(np.ndarray[double, ndim=2] bigmatrix, int rowidx, np.ndarray[double, ndim=1] bp, int idx, float offset):
    bigmatrix[rowidx][1:5] = bp[1:]
    bigmatrix[rowidx, 0] = idx
    bigmatrix[rowidx, 5] = offset