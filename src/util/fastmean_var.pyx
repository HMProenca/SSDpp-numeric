# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:53:56 2020

@author: gathu
"""

cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport pow as cpow

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cython_mean(np.ndarray[double, ndim=1] values):
    cdef uint64_t N = values.shape[0]
    cdef double x = values[0]
    cdef int i
    for i in xrange(1,N):
        x += values[i]
    x=x/N
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cython_RSS(np.ndarray[double, ndim=1] values,double mean):
    cdef uint64_t N = values.shape[0]
    cdef double x = (values[0]-mean)*(values[0]-mean)
    cdef int i
    for i in xrange(1,N):
        x += (values[i]-mean)*(values[i]-mean)
    return x


#cdef c_compute_RSS(np.ndarray[double, ndim=1] values, double meanval):
#    cdef unsigned int i,L
#    L = values.shape[0]
#    cdef double x,RSS
#    RSS = 0
#    for i in xrange(L):
#        x = cpow(values[i]-meanval,2)
#        RSS = RSS + x
#    return RSS
#
#cdef c_compute_mean(np.ndarray[double, ndim=1] values):
#    cdef unsigned int i,L
#    L = values.shape[0]
#    cdef double x,meanval
#    meanval = 0
#    for i in xrange(L):
#        x = values[i]
#        meanval = meanval + x
#    meanval = meanval/L
#    return meanval
#
#def statistic_numeric(np.ndarray[double, ndim=1] values, np.ndarray[double, ndim=1] idx_pattern,idx_default):
#    cdef unsigned int i,L
#
#def compute_RSS(np.ndarray[double, ndim=1] values, double meanval):
#    return c_compute_RSS(values, meanval)
#
#def compute_mean(np.ndarray[double, ndim=1] values):
#    return c_compute_mean(values)