#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright notice
# ----------------
#
# Copyright (C) 2013-2014 Daniel Jung
# Contact: djungbremen@gmail.com
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
"""Calculate Chebychev moments.

(At least) three kinds of moments can be distinguished:
- expectation values of the form <a|T_n(H)|b>
- expectation values with a==b, <a|T_n(H)|a> (able to yield 2 moments per loop)
- traces Tr[A*T_n(H)] (do averaging in high-level module using the above?)

Right now, only the second kind <a|T_n(H)|a> is implemented, which is needed
to calculate the local density of states (LDOS)."""
__created__ = '2012-08-06'
__modified__ = '2012-09-05'
import cython
import numpy
import scipy.sparse
cimport libc.math
cimport libc.stdlib
cimport numpy
from cython.parallel import *
cimport openmp
import misc


# to do
#def trace(...)
#def expec2(...)
#def _expec_csr_real(indices, indptr, data, rank, state, limit, out=None)
#def _expec2_csr_real(indices, indptr, data, state1, state2, limit, ...)
#def _trace_csr_real(indices, indptr, data, randstates, limit, ...)
# etc.


def expec(mat, state, limit, omp=False, num_threads=None, out=None):
    """expec(mat, state, limit, omp=False, num_threads=None, out=None)

    Calculate Chebychev moments for expectation values of the form <a|T_n(H)|a>
    for a certain state vector |a> "state" and a Hamiltonian matrix H "mat".
    Specify required number of moments "limit" (truncation limit). If "out" is
    not None, store the results in "out" instead of returning them.

    It is recommended to provide the matrix in the CSR sparse matrix format,
    since this is the best and fastest format for the matrix-vector
    multiplication which is the very heart of every KPM algorithm.

    This function only delegates the work to the corresponding low-level
    functions, all beginning with "_expec_". If no optimal low-level function
    is found for the given datatype, the plain Python implementation "_expec"
    is used (TO DO).

    If "omp" is True, use available OpenMP-parallelized variants of the
    algorithms. If "num_threads" is not None, set the number of threads
    accordingly. If "num_threads" is smaller than 1, determine the number of
    processor cores automatically."""
    __created__ = '2012-08-17'
    __modified__ = '2012-08-19'

    # prepare
    if omp:
        misc.set_num_threads(num_threads)
    limit = int(limit)
    if limit < 1:
        raise ValueError('bad truncation limit, expecting positive integer')
    #if not isinstance(mat, scipy.sparse.csr_matrix): #base.spmatrix
        #raise TypeError, 'bad matrix format, expecting CSR sparse matrix'
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('bad matrix shape, expecting square matrix')
    state = numpy.array(state)
    if state.ndim != 1:
        raise ValueError('bad state, expecting 1D array')

    # choose appropriate low-level function
    if isinstance(mat, scipy.sparse.base.spmatrix):
        # sparse matrix given
        if isinstance(mat, scipy.sparse.csr_matrix):
            # CSR sparse matrix given
            if mat.dtype is numpy.dtype(float) \
                    and state.dtype is numpy.dtype(float):
                state = state.astype(float)
                func = _expec_csr_real_omp if omp else _expec_csr_real
            elif mat.dtype is numpy.dtype(complex) \
                    or state.dtype is numpy.dtype(complex):
                state = state.astype(complex)
                mat = mat.astype(complex)  # write additional variants
                func = _expec_csr_complex_omp if omp else _expec_csr_complex
            else:
                raise ValueError('datatype not supported: %s' % mat.dtype)
            return func(mat.indices, mat.indptr, mat.data, state, limit,
                        out=out)
        else:
            # use plain Python algorithm for other sparse matrix formats
            return _expec_sparse(mat, state, limit, out=out)
    elif isinstance(mat, numpy.ndarray) or isinstance(mat, numpy.matrix):
        # dense matrix given
        return _expec_dense(mat, state, limit, out=out)
    else:
        raise ValueError('unknown matrix format')


#=====================#
# Low-level functions #
#=====================#


def _expec_dense(object mat, object state, int limit, out=None):
    __created__ = '2012-08-19'
    raise NotImplementedError('please use CSR sparse matrix')


def _expec_sparse(object mat, object state, int limit, out=None):
    __created__ = '2012-08-19'
    raise NotImplementedError('please use CSR sparse matrix')


@cython.cdivision(True)
@cython.boundscheck(False)
def _expec_csr_real(numpy.ndarray[int, mode="c"] indices,
                    numpy.ndarray[int, mode="c"] indptr,
                    numpy.ndarray[double, mode="c"] data,
                    numpy.ndarray[double, mode="c"] state,
                    int limit, numpy.ndarray[double, mode="c"] out=None):
    """_expec_csr_real(numpy.ndarray[int, mode="c"]    indices,
                    numpy.ndarray[int, mode="c"]    indptr,
                    numpy.ndarray[double, mode="c"] data,
                    numpy.ndarray[double, mode="c"] state,
                    int                             limit,
                    numpy.ndarray[double, mode="c"] out=None)

    Calculate real Chebychev moments for expectation values of the form
    <a|T_n(H)|a> for a real state vector |a> "state" and a real CSR matrix H,
    defined by column indices "indices", row pointer "indptr" and non-zero
    values "data". Specify required number of moments "limit" (truncation
    limit). If "out" is not None, store the results in "out" instead of
    returning them.

    """
    __created__ = '2012-08-19'
    __modified__ = '2012-09-05'

    # prepare
    cdef int rank = len(indptr)-1
    if len(state) != rank:
        raise ValueError('bad state, must be equal to the rank of the matrix')
    if limit < 1:
        raise ValueError('bad truncation limit, expecting positive integer')
    do_return = out is None
    if out is None:
        out = numpy.empty(limit, dtype=float)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if not indices.flags['C_CONTIGUOUS']:
        indices = numpy.ascontiguousarray(indices)
    if not indptr.flags['C_CONTIGUOUS']:
        indptr = numpy.ascontiguousarray(indptr)
    if not data.flags['C_CONTIGUOUS']:
        data = numpy.ascontiguousarray(data)
    if not state.flags['C_CONTIGUOUS']:
        state = numpy.ascontiguousarray(state)

    # calculate
    cdef:
        int *ind = <int*>indices.data
        int *ptr = <int*>indptr.data
        double *dat = <double*>data.data
        double *mom = <double*>out.data  # Py_mom.data
        double *sta = <double*>state.data
        double *phi0 = <double*>libc.stdlib.malloc(sizeof(double)*rank)
        double *phi1 = <double*>libc.stdlib.malloc(sizeof(double)*rank)
        double *phi2 = <double*>libc.stdlib.malloc(sizeof(double)*rank)
        double m0, m1  # temporary reduction variables
        int    i, j, k
    try:
        # calculate first two orders of the Chebychev expansion
        for k in range(rank):
            phi0[k] = sta[k]
        mom[0] = dot_real(phi0, phi0, rank)
        matvec_csr_real(ind, ptr, dat, rank, phi0, phi1)
        mom[1] = dot_real(phi1, phi0, rank)

        # main iteration loop
        for i in range(1, limit/2):
            # initialize temporary reduction variables
            m0 = 0
            m1 = 0

            for k in range(rank):
                # CSR-matrix-vector multiplication
                phi2[k] = 0
                for j in range(ptr[k], ptr[k+1]):
                    phi2[k] += 2*dat[j]*phi1[ind[j]]
                phi2[k] -= phi0[k]

                # calculate Chebychev moments, part 1
                # use temporary reduction variables so that Cython/OpenMP
                # understands
                m0 += 2*phi1[k]*phi1[k]  # reduction for mom[i*2]
                m1 += 2*phi2[k]*phi1[k]  # reduction for mom[i*2+1]

            # calculate Chebychev moments, part 2
            mom[i*2] = m0-mom[0]
            mom[i*2+1] = m1-mom[1]

            # shift states circular
            shift3_real(&phi0, &phi1, &phi2)

        if limit % 2 == 1:
            # calculate last Chebychev moment in case of an odd truncation
            # number
            mom[limit-1] = 2*dot_real(phi1, phi1, rank)-mom[0]

    finally:
        # clean up memory
        libc.stdlib.free(phi0)
        libc.stdlib.free(phi1)
        libc.stdlib.free(phi2)

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _expec_csr_real_omp(numpy.ndarray[int, mode="c"] indices,
                        numpy.ndarray[int, mode="c"] indptr,
                        numpy.ndarray[double, mode="c"] data,
                        numpy.ndarray[double, mode="c"] state,
                        int limit, numpy.ndarray[double, mode="c"] out=None):
    """_expec_csr_real_omp(numpy.ndarray[int, mode="c"]    indices,
                        numpy.ndarray[int, mode="c"]    indptr,
                        numpy.ndarray[double, mode="c"] data,
                        numpy.ndarray[double, mode="c"] state,
                        int                             limit,
                        numpy.ndarray[double, mode="c"] out=None)

    Calculate real Chebychev moments for expectation values of the form
    <a|T_n(H)|a> for a real state vector |a> "state" and a real CSR matrix H,
    defined by column indices "indices", row pointer "indptr" and non-zero
    values "data". Specify required number of moments "limit" (truncation
    limit). If "out" is not None, store the results in "out" instead of
    returning them.

    This is the OpenMP version of the function, using parallel for-loops."""
    __created__ = '2012-08-18'
    __modified__ = '2012-09-05'

    # prepare
    cdef int rank = len(indptr)-1
    if len(state) != rank:
        raise ValueError('bad state, must be equal to the rank of the matrix')
    if limit < 1:
        raise ValueError('bad truncation limit, expecting positive integer')
    do_return = out is None
    if out is None:
        out = numpy.empty(limit, dtype=float)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if not indices.flags['C_CONTIGUOUS']:
        indices = numpy.ascontiguousarray(indices)
    if not indptr.flags['C_CONTIGUOUS']:
        indptr = numpy.ascontiguousarray(indptr)
    if not data.flags['C_CONTIGUOUS']:
        data = numpy.ascontiguousarray(data)
    if not state.flags['C_CONTIGUOUS']:
        state = numpy.ascontiguousarray(state)

    # calculate
    cdef:
        int *ind = <int*>indices.data
        int *ptr = <int*>indptr.data
        double *dat = <double*>data.data
        double *mom = <double*>out.data  # Py_mom.data
        double *sta = <double*>state.data
        double *phi0 = <double*>libc.stdlib.malloc(sizeof(double)*rank)
        double *phi1 = <double*>libc.stdlib.malloc(sizeof(double)*rank)
        double *phi2 = <double*>libc.stdlib.malloc(sizeof(double)*rank)
        double m0, m1  # temporary reduction variables
        int    i, j, k
    try:
        # calculate first two orders of the Chebychev expansion
        for k in prange(rank, nogil=True):
            phi0[k] = sta[k]
        mom[0] = dot_real_omp(phi0, phi0, rank)
        matvec_csr_real_omp(ind, ptr, dat, rank, phi0, phi1)
        mom[1] = dot_real_omp(phi1, phi0, rank)

        # main iteration loop
        for i in range(1, limit/2):
            # initialize temporary reduction variables
            m0 = 0
            m1 = 0

            for k in prange(rank, nogil=True):
                # CSR-matrix-vector multiplication
                phi2[k] = 0
                for j in range(ptr[k], ptr[k+1]):
                    phi2[k] += 2*dat[j]*phi1[ind[j]]
                phi2[k] -= phi0[k]

                # calculate Chebychev moments, part 1
                # use temporary reduction variables so that Cython/OpenMP
                # understands
                m0 += 2*phi1[k]*phi1[k]  # reduction for mom[i*2]
                m1 += 2*phi2[k]*phi1[k]  # reduction for mom[i*2+1]

            # calculate Chebychev moments, part 2
            mom[i*2] = m0-mom[0]
            mom[i*2+1] = m1-mom[1]

            # shift states circular
            shift3_real(&phi0, &phi1, &phi2)

        if limit % 2 == 1:
            # calculate last Chebychev moment in case of an odd truncation
            # number
            mom[limit-1] = 2*dot_real_omp(phi1, phi1, rank)-mom[0]

    finally:
        # clean up memory
        libc.stdlib.free(phi0)
        libc.stdlib.free(phi1)
        libc.stdlib.free(phi2)

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _expec_csr_complex(numpy.ndarray[int, mode="c"] indices,
                       numpy.ndarray[int, mode="c"] indptr,
                       numpy.ndarray[complex, mode="c"] data,
                       numpy.ndarray[complex, mode="c"] state,
                       int limit, numpy.ndarray[complex, mode="c"] out=None):
    """_expec_csr_complex(numpy.ndarray[int, mode="c"]     indices,
                        numpy.ndarray[int, mode="c"]     indptr,
                        numpy.ndarray[complex, mode="c"] data,
                        numpy.ndarray[complex, mode="c"] state,
                        int                              limit,
                        numpy.ndarray[complex, mode="c"] out=None)

    Calculate complex Chebychev moments for expectation values of the form
    <a|T_n(H)|a> for a complex state vector |a> "state" and a complex CSR
    matrix H, defined by column indices "indices", row pointer "indptr" and
    complex non-zero values "data". Specify required number of moments "limit"
    (truncation limit). If "out" is not None, store the results in "out"
    instead of returning them.

    """
    __created__ = '2012-08-19'
    __modified__ = '2012-09-05'

    # prepare
    cdef int rank = len(indptr)-1
    if len(state) != rank:
        raise ValueError('bad state, must be equal to the rank of the matrix')
    if limit < 1:
        raise ValueError('bad truncation limit, expecting positive integer')
    do_return = out is None
    if out is None:
        out = numpy.empty(limit, dtype=complex)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if not indices.flags['C_CONTIGUOUS']:
        indices = numpy.ascontiguousarray(indices)
    if not indptr.flags['C_CONTIGUOUS']:
        indptr = numpy.ascontiguousarray(indptr)
    if not data.flags['C_CONTIGUOUS']:
        data = numpy.ascontiguousarray(data)
    if not state.flags['C_CONTIGUOUS']:
        state = numpy.ascontiguousarray(state)

    # calculate
    cdef:
        int *ind = <int*>indices.data
        int *ptr = <int*>indptr.data
        complex *dat = <complex*>data.data
        complex *mom = <complex*>out.data
        complex *sta = <complex*>state.data
        complex *phi0 = <complex*>libc.stdlib.malloc(sizeof(complex)*rank)
        complex *phi1 = <complex*>libc.stdlib.malloc(sizeof(complex)*rank)
        complex *phi2 = <complex*>libc.stdlib.malloc(sizeof(complex)*rank)
        complex m0, m1  # temporary reduction variables
        int     i, j, k
    try:
        # calculate first two orders of the Chebychev expansion
        for k in range(rank):
            phi0[k] = sta[k]
        mom[0] = dot_complex(phi0, phi0, rank)
        matvec_csr_complex(ind, ptr, dat, rank, phi0, phi1)
        mom[1] = dot_complex(phi0, phi1, rank)

        # main iteration loop
        for i in range(1, limit/2):
            # initialize temporary reduction variables
            m0 = 0
            m1 = 0

            for k in range(rank):
                # CSR-matrix-vector multiplication
                phi2[k] = 0
                for j in range(ptr[k], ptr[k+1]):
                    phi2[k] += 2*dat[j]*phi1[ind[j]]
                phi2[k] -= phi0[k]

                # calculate Chebychev moments, part 1
                # use temporary reduction variables so that Cython/OpenMP
                # understands
                m0 += 2*phi1[k].conjugate()*phi1[k]  # reduction for mom[i*2]
                m1 += 2*phi2[k].conjugate()*phi1[k]  # reduction for mom[i*2+1]

            # calculate Chebychev moments, part 2
            mom[i*2] = m0-mom[0]
            mom[i*2+1] = m1-mom[1]

            # shift states circular
            shift3_complex(&phi0, &phi1, &phi2)

        if limit % 2 == 1:
            # calculate last Chebychev moment in case of an odd truncation
            # number
            mom[limit-1] = 2*dot_complex(phi1, phi1, rank)-mom[0]

    finally:
        # clean up memory
        libc.stdlib.free(phi0)
        libc.stdlib.free(phi1)
        libc.stdlib.free(phi2)

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _expec_csr_complex_omp(numpy.ndarray[int, mode="c"] indices,
                           numpy.ndarray[int, mode="c"] indptr,
                           numpy.ndarray[complex, mode="c"] data,
                           numpy.ndarray[complex, mode="c"] state,
                           int limit,
                           numpy.ndarray[complex, mode="c"] out=None):
    """_expec_csr_complex_omp(numpy.ndarray[int, mode="c"]     indices,
                            numpy.ndarray[int, mode="c"]     indptr,
                            numpy.ndarray[complex, mode="c"] data,
                            numpy.ndarray[complex, mode="c"] state,
                            int                              limit,
                            numpy.ndarray[complex, mode="c"] out=None)

    Calculate complex Chebychev moments for expectation values of the form
    <a|T_n(H)|a> for a complex state vector |a> "state" and a complex CSR
    matrix H, defined by column indices "indices", row pointer "indptr" and
    complex non-zero values "data". Specify required number of moments "limit"
    (truncation limit). If "out" is not None, store the results in "out"
    instead of returning them.

    This is the OpenMP version of the function, using parallel for-loops."""
    __created__ = '2012-08-19'
    __modified__ = '2012-09-05'

    # prepare
    cdef int rank = len(indptr)-1
    if len(state) != rank:
        raise ValueError('bad state, must be equal to the rank of the matrix')
    if limit < 1:
        raise ValueError('bad truncation limit, expecting positive integer')
    do_return = out is None
    if out is None:
        out = numpy.empty(limit, dtype=complex)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if not indices.flags['C_CONTIGUOUS']:
        indices = numpy.ascontiguousarray(indices)
    if not indptr.flags['C_CONTIGUOUS']:
        indptr = numpy.ascontiguousarray(indptr)
    if not data.flags['C_CONTIGUOUS']:
        data = numpy.ascontiguousarray(data)
    if not state.flags['C_CONTIGUOUS']:
        state = numpy.ascontiguousarray(state)

    # calculate
    cdef:
        int *ind = <int*>indices.data
        int *ptr = <int*>indptr.data
        complex *dat = <complex*>data.data
        complex *mom = <complex*>out.data
        complex *sta = <complex*>state.data
        complex *phi0 = <complex*>libc.stdlib.malloc(sizeof(complex)*rank)
        complex *phi1 = <complex*>libc.stdlib.malloc(sizeof(complex)*rank)
        complex *phi2 = <complex*>libc.stdlib.malloc(sizeof(complex)*rank)
        double  re0, im0, re1, im1  # temporary reduction variables
        complex m0, m1
        int     i, j, k
    try:
        # calculate first two orders of the Chebychev expansion
        for k in prange(rank, nogil=True):
            phi0[k] = sta[k]
        mom[0] = dot_complex_omp(phi0, phi0, rank)
        matvec_csr_complex_omp(ind, ptr, dat, rank, phi0, phi1)
        mom[1] = dot_complex_omp(phi0, phi1, rank)

        # main iteration loop
        for i in range(1, limit/2):
            # initialize temporary reduction variables
            re0 = 0
            im0 = 0
            re1 = 0
            im1 = 0

            for k in prange(rank, nogil=True):
                # CSR-matrix-vector multiplication
                phi2[k] = 0
                for j in range(ptr[k], ptr[k+1]):
                    phi2[k] += 2*dat[j]*phi1[ind[j]]
                phi2[k] -= phi0[k]

                # calculate Chebychev moments, part 1
                m0 = 2*phi1[k].conjugate()*phi1[k]
                m1 = 2*phi2[k].conjugate()*phi1[k]

                # use temporary reduction variables so that Cython/OpenMP
                # understands
                re0 += m0.real  # reduction for mom[i*2]
                im0 += m0.imag
                re1 += m1.real  # reduction for mom[i*2+1]
                im1 += m1.imag

            # calculate Chebychev moments, part 2
            mom[i*2] = complex(re0, im0)-mom[0]
            mom[i*2+1] = complex(re1, im1)-mom[1]

            # shift states circular
            shift3_complex(&phi0, &phi1, &phi2)

        if limit % 2 == 1:
            # calculate last Chebychev moment in case of an odd truncation
            # number
            mom[limit-1] = 2*dot_complex_omp(phi1, phi1, rank)-mom[0]

    finally:
        # clean up memory
        libc.stdlib.free(phi0)
        libc.stdlib.free(phi1)
        libc.stdlib.free(phi2)

    # return
    return <object>out if do_return else None


#===========#
# Utilities #
#===========#


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline double dot_real(double *a, double *b, int length):
    """double dot_real(double *a, double *b, int length)

    Calcualte dot product of the two given real vectors "a" and "b" with the
    given length "length"."""
    # 2012-08-17
    # based on tb.kpm.core.dot, developed from 2011-12-08 until 2011-12-11
    cdef:
        double out = 0
        int    i
    for i in range(length):
        out += a[i]*b[i]
    return out


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline double dot_real_omp(double *a, double *b, int length):
    """double dot_real_omp(double *a, double *b, int length)

    Calcualte dot product of the two given real vectors "a" and "b" with the
    given length "length", using OpenMP parallelization."""
    # 2012-08-17
    # former tb.kpm.core.dot, developed from 2011-12-08 until 2011-12-11
    cdef:
        double out = 0
        int    i
    for i in prange(length, nogil=True):
        out += a[i]*b[i]
    return out


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline complex dot_complex(complex *a, complex *b, int length):
    """complex dot_complex(complex *a, complex *b, int length)

    Calcualte dot product of the two complex vectors "a" and "b" with the
    given length "length"."""
    # 2012-08-17
    # based on tb.kpm.core.dot, developed from 2011-12-08 until 2011-12-11
    cdef:
        complex out = 0
        int    i
    for i in range(length):
        out += a[i].conjugate()*b[i]
    return out


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline complex dot_complex_omp(complex *a, complex *b, int length):
    """complex dot_complex_omp(complex *a, complex *b, int length)

    Calcualte dot product of the two complex vectors "a" and "b" with the
    given length "length", using OpenMP parallelization."""
    # 2012-08-17 until 2012-08-20
    # based on tb.kpm.core.dot, developed from 2011-12-08 until 2011-12-11
    cdef:
        complex temp
        double  re = 0, im = 0
        int     i
    for i in prange(length, nogil=True):
        temp = a[i].conjugate()*b[i]
        re += temp.real
        im += temp.imag
    return complex(re, im)


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline void matvec_csr_real(int *ind, int *ptr, double *dat, int rank,
                                 double *vect, double *out):
    """matvec_csr_real(int *ind, int *ptr, double *dat, int rank,
                    double *vect, double *out)

    Multiply the given real square CSR sparse matrix, defined by row indices
    "ind", column pointer "ptr", data values "dat" and rank "rank", with the
    given vector "vect". Store the resulting vector in "out" (overwrite).

    """
    # 2012-08-17
    # based on tb.kpm.core.matvec, developed from 2011-12-06 until 2011-12-11
    cdef int i, j
    for i in range(rank):
        out[i] = 0.
        for j in range(ptr[i], ptr[i+1]):
            out[i] += dat[j]*vect[ind[j]]


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline void matvec_csr_real_omp(int *ind, int *ptr, double *dat, int rank,
                                     double *vect, double *out):
    """matvec_csr_real_omp(int *ind, int *ptr, double *dat, int rank,
                        double *vect, double *out)

    Multiply the given real square CSR sparse matrix, defined by row indices
    "ind", column pointer "ptr", data values "dat" and rank "rank", with the
    given vector "vect". Store the resulting vector in "out" (overwrite).

    This is the OpenMP variant of the function, using a parallel for-loop."""
    # 2012-08-17
    # former tb.kpm.core.matvec, developed from 2011-12-06 until 2011-12-11
    cdef int i, j
    for i in prange(rank, nogil=True):
        out[i] = 0
        for j in range(ptr[i], ptr[i+1]):
            out[i] += dat[j]*vect[ind[j]]


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline void matvec_csr_complex(int *ind, int *ptr, complex *dat, int rank,
                                    complex *vect, complex *out):
    """matvec_csr_complex(int *ind, int *ptr, complex *dat, int rank,
                        complex *vect, complex *out)

    Multiply the given complex square CSR sparse matrix, defined by row indices
    "ind", column pointer "ptr", data values "dat" and rank "rank", with the
    given complex vector "vect". Store the resulting vector in "out"
    (overwrite).

    """
    # 2012-08-17
    # based on tb.kpm.core.matvec, developed from 2011-12-06 until 2011-12-11
    cdef int i, j
    for i in range(rank):
        out[i] = 0.
        for j in range(ptr[i], ptr[i+1]):
            out[i] += dat[j]*vect[ind[j]]


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline void matvec_csr_complex_omp(int *ind, int *ptr, complex *dat, int
                                        rank, complex *vect, complex *out):
    """matvec_csr_complex_omp(int *ind, int *ptr, complex *dat, int rank,
                            complex *vect, complex *out)

    Multiply the given complex square CSR sparse matrix, defined by row indices
    "ind", column pointer "ptr", data values "dat" and rank "rank", with the
    given complex vector "vect". Store the resulting vector in "out"
    (overwrite).

    This is the OpenMP variant of the function, using a parallel for-loop."""
    # 2012-08-17
    # based on tb.kpm.core.matvec, developed from 2011-12-06 until 2011-12-11
    cdef:
        int i, j
    for i in prange(rank, nogil=True):
        out[i] = 0
        for j in range(ptr[i], ptr[i+1]):
            out[i] += dat[j]*vect[ind[j]]


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline void shift3_real(double **a, double **b, double **c):
    """Permute the three given double pointers circular. Call with "&" (address
    of each pointer)."""
    # former tb.kpm.core.shift3, developed from 2011-12-06 until 2011-12-11
    cdef double *temp = a[0]
    a[0] = b[0]
    b[0] = c[0]
    c[0] = temp


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline void shift3_complex(complex **a, complex **b, complex **c):

    """Permute the three given complex pointers circular. Call with "&"
    (address of each pointer).

    """
    # based on tb.kpm.core.shift3, developed from 2011-12-06 until 2011-12-11
    cdef complex *temp = a[0]
    a[0] = b[0]
    b[0] = c[0]
    c[0] = temp


cdef print_array(double *arr, int length):
    """Debugging utility to print a complete 1D C array to stdout."""
    __created__ = '2012-08-19'
    cdef int i
    for i in range(length):
        print arr[i],
    print


cdef print_array_int(int *arr, int length):
    """Debugging utility to print a complete 1D C array to stdout."""
    __created__ = '2012-09-05'
    cdef int i
    for i in range(length):
        print arr[i],
    print
