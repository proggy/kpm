#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright notice
# ----------------
#
# Copyright (C) 2013-2023 Daniel Jung
# Contact: proggy-contact@mailbox.org
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
"""Reconstruct the target function, using different methods.
"""

import cython
import numpy
cimport libc.math
cimport libc.stdlib
cimport numpy
cimport openmp
from cython.parallel import *
from kpm import misc


def select(string, many=False):
    """select(string, many=False)

    Select one of the reconstruction methods defined in this module by the
    given *string*. For example, this could be an option string from a command
    line option.
    """

    string = str(string)
    if string and ('std'.startswith(string)
                   or 'standard'.startswith(string)
                   or 'default'.startswith(string)):
        return std_many if many else std
    elif string and ('dct'.startswith(string)
                     or 'discrete cosine transformation'.startswith(string)
                     or 'cosine transformation'.startswith(string)):
        return dct_many if many else dct
    #elif string and ('fct'.startswith(string) \
                    #or 'fast cosine transformation'.startswith(string)):
        #return fct
    #elif string and ('dft'.startswith(string) \
                    #or 'discrete fourier transformation'.startswith(string) \
                    #or 'fourier transformation'.startswith(string)):
        #return dct
    #elif string and ('fft'.startswith(string) \
                    #or 'fast fourier transformation'.startswith(string)):
        #return fft
    else:
        raise ValueError('reconstruction method not found: %s' % string)


def all():
    """all()

    Return a list of all reconstruction methods defined in this module.
    """
    funclist = [std, dct]  # fct, dft, fft
    funclist.sort()
    return funclist


#================================#
# Standard reconstruction method #
#================================#


def std(moments, disc, varmom=None, omp=True, num_threads=None, out=None):
    """std(moments, disc=None, varmom=None, limits=None, omp=True,
        num_threads=None, out=None)

    Reconstruct target function using the "naive" Chebychev series
    expansion. A user-defined x-axis discretization *disc* has to be specified.

    This function only delegates the work to the corresponding low-level
    functions, all beginning with "_std_". If no optimal low-level function is
    found for the given datatype, the plain Python implementation "_std"
    is used.

    If *omp* is *True*, use available OpenMP-parallelized variants of the
    algorithms. If *num_threads* is not *None*, set the number of threads
    accordingly. If *num_threads* is smaller than 1, determine and use the
    number of processor cores.

    If *varmom* is given, use a variable moments approach. *varmom* has to have
    the same shape as *disc*, specifying a truncation limit for each
    discretization step. The values of *varmom* must not be greater than the
    length of *moments*.
    """
    if type(moments) is not numpy.ndarray:
        moments = numpy.ascontiguousarray(moments)
    #if disc is None:
        #raise ValueError, 'no discretization given (disc)'
    if type(disc) is not numpy.ndarray:  # disc is not None and
        disc = numpy.ascontiguousarray(disc)
    if varmom is not None and type(varmom) is not numpy.ndarray:
        varmom = numpy.ascontiguousarray(varmom)
    if omp:
        misc.set_num_threads(num_threads)
    if varmom:
        if moments.dtype is numpy.dtype(float):
            if omp:
                raise NotImplementedError
            func = _std_real_vm  # _std_real_omp_vm if omp else
        elif moments.dtype is numpy.dtype(complex):
            raise NotImplementedError
            #func = _std_complex_omp_vm if omp else _std_complex_vm
        else:
            raise NotImplementedError
            #func = _std_vm
        return func(moments, disc, varmom, out=out)
    else:
        if moments.dtype is numpy.dtype(float):
            func = _std_real_omp if omp else _std_real
        elif moments.dtype is numpy.dtype(complex):
            func = _std_complex_omp if omp else _std_complex
        else:
            func = _std
        return func(moments, disc, out=out)


def std_many(moments, disc, limits=None, omp=True, num_threads=None,
             out=None):
    """std(moments, disc, limits=None, omp=True, num_threads=None, out=None)

    Reconstruct many versions of the target function at once, each for a
    different truncation limit. Use the "naive" Chebychev series expansion.  A
    user-defined x-axis discretization *disc* has to be specified. If *limits*
    is *None*, default to the number of moments.

    This function only delegates the work to the corresponding low-level
    functions, all beginning with "_std_". If no optimal low-level function is
    found for the given datatype, the plain Python implementation "_std"
    is used.

    If *omp* is *True*, use available OpenMP-parallelized variants of the
    algorithms. If *num_threads* is not *None*, set the number of threads
    accordingly. If *num_threads* is smaller than 1, determine and use the
    number of processor cores.
    """
    if type(moments) is not numpy.ndarray:
        moments = numpy.ascontiguousarray(moments)
    if type(disc) is not numpy.ndarray:
        disc = numpy.ascontiguousarray(disc)
    if not numpy.all(disc) < 1. or not numpy.all(disc) > -1.:
        raise ValueError('disc values must be from the open interval (-1, 1)')
    if limits is None:
        limits = [len(moments), ]
    limits = numpy.array(limits, dtype=numpy.int32)
    if omp:
        misc.set_num_threads(num_threads)

    if moments.dtype is numpy.dtype(float):
        func = _std_real_omp_many if omp else _std_real_many
    elif moments.dtype is numpy.dtype(complex):
        func = _std_complex_omp_many if omp else _std_complex_many
    else:
        func = _std_many
    return func(moments, disc, limits, out=out)


def _std(numpy.ndarray[double, mode='c'] moments,
         numpy.ndarray[double, mode='c'] disc,
         numpy.ndarray[double, mode='c'] out=None):
    """_std(numpy.ndarray[double, mode='c'] moments,
        numpy.ndarray[double, mode='c'] disc,
        numpy.ndarray[double, mode='c'] out=None)

    Reconstruct target function using the given discretization "disc". If out
    is None, return the results, otherwise save them to the given array.

    This is the pure Python version of this function, using only plain
    numpy-ndarray operations. The advantage is that it is independent of the
    given datatype.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if not disc.flags['C_CONTIGUOUS']:
        disc = numpy.ascontiguousarray(disc)
    if out is None:
        out = numpy.empty(len(disc), dtype=float)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if disc.ndim != 1:
        raise ValueError('bad discretization, expecting 1D array')
    if out.shape[0] != disc.shape[0] or out.ndim != 1:
        raise ValueError('bad output array: wrong shape')

    # calculate
    raise NotImplementedError
    #limit = len(moments)
    #j = numpy.arange(limit)
    #out[0] = moments[0]
    #out[1:] = 2*moments   *   numpy.cos(j   *    numpy.acos(disc))[None, :]
    #out /= numpy.pi*numpy.sqrt(1-disc**2)

    # return
    #return out if do_return else None


def _std_many(numpy.ndarray[double, mode='c']         moments,
              numpy.ndarray[double, mode='c']         disc,
              numpy.ndarray[int, mode='c']            limits,
              numpy.ndarray[double, ndim=2, mode='c'] out=None):
    """_std_many(numpy.ndarray[double, mode='c']         moments,
                numpy.ndarray[double, mode='c']         disc,
                numpy.ndarray[int, mode='c']            limits,
                numpy.ndarray[double, ndim=2, mode='c'] out=None

    Reconstruct target functions using the given discretization "disc" and
    the given truncation limits "limits". If out is None, return the results,
    otherwise save them to the given array.

    This is the pure Python version of this function, using only plain
    numpy-ndarray operations. The advantage is that it is independent of the
    given datatype.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if not disc.flags['C_CONTIGUOUS']:
        disc = numpy.ascontiguousarray(disc)
    if not limits.flags['C_CONTIGUOUS']:
        limits = numpy.ascontiguousarray(limits)
    if out is None:
        out = numpy.empty((len(limits), len(disc)), dtype=float)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if disc.ndim != 1:
        raise ValueError('bad discretization, expecting 1D array')
    if out.shape[0] != limits.shape[0] or out.shape[1] != disc.shape[0] \
            or out.ndim != 2:
        raise ValueError('bad output array: wrong shape')

    # calculate
    raise NotImplementedError
    #limit = len(moments)
    #j = numpy.arange(limit)
    #out[0] = moments[0]
    #out[1:] = 2*moments   *   numpy.cos(j   *    numpy.acos(disc))[None, :]
    #out /= numpy.pi*numpy.sqrt(1-disc**2)

    # return
    #return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _std_real(numpy.ndarray[double, mode='c'] moments,
              numpy.ndarray[double, mode='c'] disc,
              numpy.ndarray[double, mode='c'] out=None):
    """_std_real(numpy.ndarray[double, mode='c'] moments,
              numpy.ndarray[double, mode='c'] disc,
              numpy.ndarray[double, mode='c'] out=None)

    Reconstruct target function using the given discretization "disc" and the
    real-valued Chebychev moments "moments". If out is None, return the
    results, otherwise save them to the given array.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if not disc.flags['C_CONTIGUOUS']:
        disc = numpy.ascontiguousarray(disc)
    if out is None:
        out = numpy.empty(len(disc), dtype=float)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if disc.ndim != 1:
        raise ValueError('bad discretization, expecting 1D array')
    if out.shape[0] != disc.shape[0] or out.ndim != 1:
        raise ValueError('bad output array: wrong shape')

    # calculate
    cdef:
        int limit = len(moments)
        int i, j
        int ndisc = len(disc)
        double *mom = <double*>moments.data
        double *dis = <double*>disc.data
        double *o = <double*>out.data
    for i in range(ndisc):
        o[i] = mom[0]
        for j in range(1, limit):
            o[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
        o[i] /= libc.math.M_PI*libc.math.sqrt(1-dis[i]*dis[i])

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _std_real_vm(numpy.ndarray[double, mode='c'] moments,
              numpy.ndarray[double, mode='c'] disc,
              numpy.ndarray[int, mode='c'] varmom,
              numpy.ndarray[double, mode='c'] out=None):
    """_std_real_vm(numpy.ndarray[double, mode='c'] moments,
                 numpy.ndarray[double, mode='c'] disc,
                 numpy.ndarray[int, mode='c'] varmom,
                 numpy.ndarray[double, mode='c'] out=None)

    Reconstruct target function using the given discretization "disc" and the
    real-valued Chebychev moments "moments". A variable-moment approach is
    commenced, the number of moments for each discretization step is given by
    "varmom". If out is None, return the results, otherwise save them to the
    given array.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if not disc.flags['C_CONTIGUOUS']:
        disc = numpy.ascontiguousarray(disc)
    if not varmom.flags['C_CONTIGUOUS']:
        varmom = numpy.ascontiguousarray(varmom)
    if out is None:
        out = numpy.empty(len(disc), dtype=float)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if disc.ndim != 1:
        raise ValueError('bad discretization, expecting 1D array')
    if varmom.ndim != 1 or len(varmom) != len(disc):
        raise ValueError('shape of "varmom" inconsistent with that of "disc"')
    if out.shape[0] != disc.shape[0] or out.ndim != 1:
        raise ValueError('bad output array: wrong shape')
    if numpy.any(varmom > len(moments)):
        raise ValueError('not enough moments given')

    # calculate
    cdef:
        int limit = len(moments)
        int i, j
        int ndisc = len(disc)
        double *mom = <double*>moments.data
        double *dis = <double*>disc.data
        int *vm = <int*>varmom.data
        double *o = <double*>out.data
    for i in range(ndisc):
        o[i] = mom[0]
        for j in range(1, vm[i]):
            o[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
        o[i] /= libc.math.M_PI*libc.math.sqrt(1-dis[i]*dis[i])

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _std_real_many(numpy.ndarray[double, mode='c']         moments,
                   numpy.ndarray[double, mode='c']         disc,
                   numpy.ndarray[int, mode='c']            limits,
                   numpy.ndarray[double, ndim=2, mode='c'] out=None):
    """_std_real_many(numpy.ndarray[double, mode='c']         moments,
                    numpy.ndarray[double, mode='c']         disc,
                    numpy.ndarray[int, mode='c']            limits,
                    numpy.ndarray[double, ndim=2, mode='c'] out=None)

    Reconstruct target functions using the given discretization "disc", the
    real-valued Chebychev moments "moments" and the truncation limits "limits".
    If out is None, return the results, otherwise save them to the given
    2D-array.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if not disc.flags['C_CONTIGUOUS']:
        disc = numpy.ascontiguousarray(disc)
    if not limits.flags['C_CONTIGUOUS']:
        limits = numpy.ascontiguousarray(limits)
    if out is None:
        out = numpy.empty((len(limits), len(disc)), dtype=float)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if disc.ndim != 1:
        raise ValueError('bad discretization, expecting 1D array')
    if out.shape[0] != limits.shape[0] or out.shape[1] != disc.shape[0] \
            or out.ndim != 2:
        raise ValueError('bad output array: wrong shape')

    cdef numpy.ndarray[double, ndim=2, mode='c'] temp
    temp = numpy.empty((len(limits), len(disc)), dtype=float)

    # calculate
    cdef:
        int limit, limit1, limit2
        int i, j, k
        int ndisc = len(disc)
        int nlimits = len(limits)
        double *mom = <double*>moments.data
        double *dis = <double*>disc.data
        double *t = <double*>temp.data
        double **o = <double**>libc.stdlib.malloc(nlimits*sizeof(double*))
        double denom
    try:
        for k in range(nlimits):
            o[k] = &out[k, 0]
        for i in range(ndisc):
            denom = libc.math.M_PI*libc.math.sqrt(1-dis[i]*dis[i])

        # iterate until first truncation limit
        t[i] = mom[0]
        for j in range(1, limits[0]):
            t[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
        o[0][i] = t[i]/denom

        # iterate until each of the truncation limits
        for k in range(1, nlimits):
            limit1 = limits[k-1]
            limit2 = limits[k]
            for j in range(limit1, limit2):
                t[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
            o[k][i] = t[i]/denom

    finally:
        libc.stdlib.free(o)

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _std_real_omp(numpy.ndarray[double, mode='c'] moments,
                  numpy.ndarray[double, mode='c'] disc,
                  numpy.ndarray[double, mode='c'] out=None):
    """_std_real_omp(numpy.ndarray[double, mode='c'] moments,
                    numpy.ndarray[double, mode='c'] disc,
                    numpy.ndarray[double, mode='c'] out=None)

    Reconstruct target function using the given discretization "disc". If out
    is None, return the results, otherwise save them to the given array.

    This is the OpenMP version of the function, using a parallel for-loop.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if not disc.flags['C_CONTIGUOUS']:
        disc = numpy.ascontiguousarray(disc)
    if out is None:
        out = numpy.empty(len(disc), dtype=float)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if disc.ndim != 1:
        raise ValueError('bad discretization, expecting 1D array')
    if out.shape[0] != disc.shape[0] or out.ndim != 1:
        raise ValueError('bad output array: wrong shape')

    # calculate
    cdef:
        int limit = len(moments)
        int i, j
        int ndisc = len(disc)
        double *mom = <double*>moments.data
        double *dis = <double*>disc.data
        double *o = <double*>out.data
    for i in prange(ndisc, nogil=True):
        o[i] = mom[0]
        for j in range(1, limit):
            o[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
        o[i] /= libc.math.M_PI*libc.math.sqrt(1-dis[i]*dis[i])

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _std_real_omp_many(numpy.ndarray[double, mode='c']         moments,
                       numpy.ndarray[double, mode='c']         disc,
                       numpy.ndarray[int, mode='c']            limits,
                       numpy.ndarray[double, ndim=2, mode='c'] out=None):
    """_std_real_omp_many(numpy.ndarray[double, mode='c']         moments,
                        numpy.ndarray[double, mode='c']         disc,
                        numpy.ndarray[int, mode='c']            limits,
                        numpy.ndarray[double, ndim=2, mode='c'] out=None)

    Reconstruct target functions using the given discretization "disc", the
    real-valued Chebychev moments "moments" and the truncation limits "limits".
    If out is None, return the results, otherwise save them to the given
    2D-array.

    This is the OpenMP version of the function, using a parallel for-loop.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if not disc.flags['C_CONTIGUOUS']:
        disc = numpy.ascontiguousarray(disc)
    if not limits.flags['C_CONTIGUOUS']:
        limits = numpy.ascontiguousarray(limits)
    if out is None:
        out = numpy.empty((len(limits), len(disc)), dtype=float)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if disc.ndim != 1:
        raise ValueError('bad discretization, expecting 1D array')
    if out.shape[0] != limits.shape[0] or out.shape[1] != disc.shape[0] \
            or out.ndim != 2:
        raise ValueError('bad output array: wrong shape')

    cdef numpy.ndarray[double, ndim=2, mode='c'] temp
    temp = numpy.empty((len(limits), len(disc)), dtype=float)

    # calculate
    cdef:
        int limit, limit1, limit2
        int i, j, k
        int ndisc = len(disc)
        int nlimits = len(limits)
        double *mom = <double*>moments.data
        double *dis = <double*>disc.data
        double *t = <double*>temp.data
        double **o = <double**>libc.stdlib.malloc(nlimits*sizeof(double*))
        double denom
    try:
        for k in range(nlimits):
            o[k] = &out[k, 0]
        for i in prange(ndisc, nogil=True):
            denom = libc.math.M_PI*libc.math.sqrt(1-dis[i]*dis[i])

        # iterate until first truncation limit
        t[i] = mom[0]
        for j in range(1, limits[0]):
            t[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
        o[0][i] = t[i]/denom

        # iterate until each of the truncation limits
        for k in range(1, nlimits):
            limit1 = limits[k-1]
            limit2 = limits[k]
            for j in range(limit1, limit2):
                t[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
            o[k][i] = t[i]/denom

    finally:
        libc.stdlib.free(o)

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _std_complex(numpy.ndarray[complex, mode='c'] moments,
                 numpy.ndarray[double,  mode='c'] disc,
                 numpy.ndarray[complex, mode='c'] out=None):
    """_std_complex(numpy.ndarray[complex, mode='c'] moments,
                numpy.ndarray[double,  mode='c'] disc,
                numpy.ndarray[complex, mode='c'] out=None)

    Reconstruct target function using the given discretization "disc". If out
    is None, return the results, otherwise save them to the given array.

    This function accepts complex moments "moments", so also the result "out"
    can be complex in general.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if not disc.flags['C_CONTIGUOUS']:
        disc = numpy.ascontiguousarray(disc)
    if out is None:
        out = numpy.empty(len(disc), dtype=complex)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if disc.ndim != 1:
        raise ValueError('bad discretization, expecting 1D array')
    if out.shape[0] != disc.shape[0] or out.ndim != 1:
        raise ValueError('bad output array: wrong shape')

    # calculate
    cdef:
        int limit = len(moments)
        int i, j
        int ndisc = len(disc)
        complex *mom = <complex*>moments.data
        double *dis = <double*>disc.data
        complex *o = <complex*>out.data
    for i in range(ndisc):
        o[i] = mom[0]
        for j in range(1, limit):
            o[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
        o[i] /= libc.math.M_PI*libc.math.sqrt(1-dis[i]*dis[i])

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _std_complex_many(numpy.ndarray[complex, mode='c']         moments,
                      numpy.ndarray[double, mode='c']          disc,
                      numpy.ndarray[int, mode='c']             limits,
                      numpy.ndarray[complex, ndim=2, mode='c'] out=None):
    """_std_complex_many(numpy.ndarray[complex, mode='c']         moments,
                        numpy.ndarray[double, mode='c']          disc,
                        numpy.ndarray[int, mode='c']             limits,
                        numpy.ndarray[complex, ndim=2, mode='c'] out=None)

    Reconstruct target functions using the given discretization "disc", the
    complex-valued Chebychev moments "moments" and the truncation limits
    "limits". If out is None, return the results, otherwise save them to the
    given 2D-array.

    This function accepts complex moments "moments", so also the result "out"
    can be complex in general.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if not disc.flags['C_CONTIGUOUS']:
        disc = numpy.ascontiguousarray(disc)
    if not limits.flags['C_CONTIGUOUS']:
        limits = numpy.ascontiguousarray(limits)
    if out is None:
        out = numpy.empty((len(limits), len(disc)), dtype=complex)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if disc.ndim != 1:
        raise ValueError('bad discretization, expecting 1D array')
    if out.shape[0] != limits.shape[0] or out.shape[1] != disc.shape[0] \
            or out.ndim != 2:
        raise ValueError('bad output array: wrong shape')

    cdef numpy.ndarray[complex, ndim=2, mode='c'] temp
    temp = numpy.empty((len(limits), len(disc)), dtype=complex)

    # calculate
    cdef:
        int limit, limit1, limit2
        int i, j, k
        int ndisc = len(disc)
        int nlimits = len(limits)
        complex *mom = <complex*>moments.data
        double *dis = <double*>disc.data
        complex *t = <complex*>temp.data
        complex **o = <complex**>libc.stdlib.malloc(nlimits*sizeof(complex*))
        double denom
    try:
        for k in range(nlimits):
            o[k] = &out[k, 0]
        for i in range(ndisc):
            denom = libc.math.M_PI*libc.math.sqrt(1-dis[i]*dis[i])

        # iterate until first truncation limit
        t[i] = mom[0]
        for j in range(1, limits[0]):
            t[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
        o[0][i] = t[i]/denom

        # iterate until each of the truncation limits
        for k in range(1, nlimits):
            limit1 = limits[k-1]
            limit2 = limits[k]
            for j in range(limit1, limit2):
                t[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
            o[k][i] = t[i]/denom

    finally:
        libc.stdlib.free(o)

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _std_complex_omp(numpy.ndarray[complex, mode='c'] moments,
                     numpy.ndarray[double,  mode='c'] disc,
                     numpy.ndarray[complex, mode='c'] out=None):
    """_std_complex_omp(numpy.ndarray[complex, mode='c'] moments,
                    numpy.ndarray[double,  mode='c'] disc,
                    numpy.ndarray[complex, mode='c'] out=None)

    Reconstruct target function using the given discretization "disc". If out
    is None, return the results, otherwise save them to the given array.

    This function accepts complex moments "moments", so also the result "out"
    is generally complex.

    This is the OpenMP version of the function, using a parallel for-loop.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if not disc.flags['C_CONTIGUOUS']:
        disc = numpy.ascontiguousarray(disc)
    if out is None:
        out = numpy.empty(len(disc), dtype=complex)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if disc.ndim != 1:
        raise ValueError('bad discretization, expecting 1D array')
    if out.shape[0] != disc.shape[0] or out.ndim != 1:
        raise ValueError('bad output array: wrong shape')

    # calculate
    cdef:
        int limit = len(moments)
        int i, j
        int ndisc = len(disc)
        complex *mom = <complex*>moments.data
        double *dis = <double*>disc.data
        complex *o = <complex*>out.data
    for i in prange(ndisc, nogil=True):
        o[i] = mom[0]
        for j in range(1, limit):
            o[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
        o[i] /= libc.math.M_PI*libc.math.sqrt(1-dis[i]*dis[i])

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _std_complex_omp_many(numpy.ndarray[complex, mode='c']         moments,
                          numpy.ndarray[double, mode='c']          disc,
                          numpy.ndarray[int, mode='c']             limits,
                          numpy.ndarray[complex, ndim=2, mode='c'] out=None):
    """_std_complex_omp_many(numpy.ndarray[complex, mode='c']         moments,
                            numpy.ndarray[double, mode='c']          disc,
                            numpy.ndarray[int, mode='c']             limits,
                            numpy.ndarray[complex, ndim=2, mode='c'] out=None)

    Reconstruct target functions using the given discretization "disc", the
    complex-valued Chebychev moments "moments" and the truncation limits
    "limits". If out is None, return the results, otherwise save them to the
    given 2D-array.

    This function accepts complex moments "moments", so also the result "out"
    can be complex in general.

    This is the OpenMP version of the function, using a parallel for-loop.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if not disc.flags['C_CONTIGUOUS']:
        disc = numpy.ascontiguousarray(disc)
    if not limits.flags['C_CONTIGUOUS']:
        limits = numpy.ascontiguousarray(limits)
    if out is None:
        out = numpy.empty((len(limits), len(disc)), dtype=complex)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if disc.ndim != 1:
        raise ValueError('bad discretization, expecting 1D array')
    if out.shape[0] != limits.shape[0] or out.shape[1] != disc.shape[0] \
            or out.ndim != 2:
        raise ValueError('bad output array: wrong shape')

    cdef numpy.ndarray[complex, ndim=2, mode='c'] temp
    temp = numpy.empty((len(limits), len(disc)), dtype=complex)

    # calculate
    cdef:
        int limit, limit1, limit2
        int i, j, k
        int ndisc = len(disc)
        int nlimits = len(limits)
        complex *mom = <complex*>moments.data
        double *dis = <double*>disc.data
        complex *t = <complex*>temp.data
        complex **o = <complex**>libc.stdlib.malloc(nlimits*sizeof(complex*))
        double denom
    try:
        for k in range(nlimits):
            o[k] = &out[k, 0]
        for i in prange(ndisc, nogil=True):
            denom = libc.math.M_PI*libc.math.sqrt(1-dis[i]*dis[i])

        # iterate until first truncation limit
        t[i] = mom[0]
        for j in range(1, limits[0]):
            t[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
        o[0][i] = t[i]/denom

        # iterate until each of the truncation limits
        for k in range(1, nlimits):
            limit1 = limits[k-1]
            limit2 = limits[k]
            for j in range(limit1, limit2):
                t[i] += mom[j]*libc.math.cos(j*libc.math.acos(dis[i]))*2
            o[k][i] = t[i]/denom

    finally:
        libc.stdlib.free(o)

    # return
    return <object>out if do_return else None


#===========================#
# Discrete cosine transform #
#===========================#


def dct(moments, ndisc=None, omp=True, num_threads=None, out=None):
    """dct(moments, ndisc=None, omp=True, num_threads=None, out=None)

    Reconstruct target function using discrete cosine transformation. Use the
    function :func:`disc.cosine` from the :mod:`disc` submodule with the same
    number of discretization steps *ndisc* as here to get the corresponding
    x-axis discretization, otherwise, the returned target function values are
    not valid.

    This function only delegates the work to the corresponding low-level
    functions, all beginning with "_std_". If no optimal low-level function is
    found for the given datatype, the plain Python implementation "_std"
    is used.

    If *omp* is *True*, use available OpenMP-parallelized variants of the
    algorithms. If *num_threads* is not *None*, set the number of threads
    accordingly. If *num_threads* is smaller than 1, determine and use the
    number of processor cores.
    """
    if type(moments) is not numpy.ndarray:
        moments = numpy.ascontiguousarray(moments)
    if omp:
        misc.set_num_threads(num_threads)
    if moments.dtype is numpy.dtype(float):
        func = _dct_real_omp if omp else _dct_real
    elif moments.dtype is numpy.dtype(complex):
        func = _dct_complex_omp if omp else _dct_complex
    else:
        func = _dct
    if ndisc is None:
        ndisc = 0
    return func(moments, ndisc=ndisc, out=out)


def dct_many(moments, ndisc=None, limits=None, omp=True, num_threads=None,
             out=None):
    raise NotImplementedError


def _dct(moments, ndisc=0, out=None):
    raise NotImplementedError


@cython.cdivision(True)
@cython.boundscheck(False)
def _dct_real(numpy.ndarray[double, mode='c'] moments,
              int                             ndisc=0,
              numpy.ndarray[double, mode='c'] out=None):
    """_dct_real(numpy.ndarray[double, mode='c'] moments,
                int                             ndisc=0,
                numpy.ndarray[double, mode='c'] out=None)

    Use discrete cosine transform to reconstruct target function. Use "ndisc"
    discretization steps. If "ndisc" is smaller than 1, set ndisc=2*limit,
    where limit is the number of moments (truncation limit). If out is None,
    return the results, otherwise save them to the given array.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if ndisc < 1:
        if do_return:
            ndisc = 2*len(moments)
        else:
            ndisc = len(out)
    if out is None:
        out = numpy.empty(ndisc, dtype=float)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != ndisc or out.ndim != 1:
        raise ValueError('bad output array: wrong shape')

    # calculate
    cdef:
        int limit = len(moments)
        int i, j
        double *mom = <double*>moments.data
        double *o = <double*>out.data
        double disc, arg
    for i in range(ndisc):
        arg = libc.math.M_PI*(2*i+1)/2/ndisc
        o[i] = mom[0]
        for j in range(1, limit):
            o[i] += 2*mom[j]*libc.math.cos(j*arg)
        disc = libc.math.cos(arg)
        o[i] /= libc.math.M_PI*libc.math.sqrt(1-disc*disc)

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _dct_real_omp(numpy.ndarray[double, mode='c'] moments,
                  int                             ndisc=0,
                  numpy.ndarray[double, mode='c'] out=None):
    """_dct_real_omp(numpy.ndarray[double, mode='c'] moments,
                    int                             ndisc=0,
                    numpy.ndarray[double, mode='c'] out=None)

    Use discrete cosine transform to reconstruct target function. Use "ndisc"
    discretization steps. If "ndisc" is smaller than 1, set ndisc=2*limit,
    where limit is the number of moments (truncation limit). If out is None,
    return the results, otherwise save them to the given array.

    This is the OpenMP version of the function, using a parallel for-loop.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if ndisc < 1:
        if do_return:
            ndisc = 2*len(moments)
        else:
            ndisc = len(out)
    if out is None:
        out = numpy.empty(ndisc, dtype=float)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != ndisc or out.ndim != 1:
        raise ValueError('bad output array: wrong shape')

    # calculate
    cdef:
        int limit = len(moments)
        int i, j
        double *mom = <double*>moments.data
        double *o = <double*>out.data
        double disc
    for i in prange(ndisc, nogil=True):
        o[i] = mom[0]
        for j in range(1, limit):
            o[i] += mom[j]*libc.math.cos(j*libc.math.M_PI*(2*i+1)/2/ndisc)*2
        disc = libc.math.cos(libc.math.M_PI*(2*i+1)/2/ndisc)
        o[i] /= libc.math.M_PI*libc.math.sqrt(1-disc*disc)

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _dct_complex(numpy.ndarray[complex, mode='c'] moments,
                 int                              ndisc=0,
                 numpy.ndarray[complex, mode='c'] out=None):
    """_dct_complex(numpy.ndarray[complex, mode='c'] moments,
                int                              ndisc=0,
                numpy.ndarray[complex, mode='c'] out=None)

    Use discrete cosine transform to reconstruct target function. Use "ndisc"
    discretization steps. If "ndisc" is smaller than 1, set ndisc=2*limit,
    where limit is the number of moments (truncation limit). If out is None,
    return the results, otherwise save them to the given array.

    This function accepts complex Chebychev moments.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if ndisc < 1:
        if do_return:
            ndisc = 2*len(moments)
        else:
            ndisc = len(out)
    if out is None:
        out = numpy.empty(ndisc, dtype=complex)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != ndisc or out.ndim != 1:
        raise ValueError('bad output array: wrong shape')

    # calculate
    cdef:
        int limit = len(moments)
        int i, j
        complex *mom = <complex*>moments.data
        complex *o = <complex*>out.data
        double disc
    for i in range(ndisc):
        o[i] = mom[0]
        for j in range(1, limit):
            o[i] += mom[j]*libc.math.cos(j*libc.math.M_PI*(2*i+1)/2/ndisc)*2
        disc = libc.math.cos(libc.math.M_PI*(2*i+1)/2/ndisc)
        o[i] /= libc.math.M_PI*libc.math.sqrt(1-disc*disc)

    # return
    return <object>out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _dct_complex_omp(numpy.ndarray[complex, mode='c'] moments,
                     int                              ndisc=0,
                     numpy.ndarray[complex, mode='c'] out=None):
    """_dct_complex_omp(numpy.ndarray[complex, mode='c'] moments,
                    int                              ndisc=0,
                    numpy.ndarray[complex, mode='c'] out=None)

    Use discrete cosine transform to reconstruct target function. Use "ndisc"
    discretization steps. If "ndisc" is smaller than 1, set ndisc=2*limit,
    where limit is the number of moments (truncation limit). If out is None,
    return the results, otherwise save them to the given array.

    This function accepts complex Chebychev moments.

    This is the OpenMP version of the function, using a parallel for-loop.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if ndisc < 1:
        if do_return:
            ndisc = 2*len(moments)
        else:
            ndisc = len(out)
    if out is None:
        out = numpy.empty(ndisc, dtype=complex)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != ndisc or out.ndim != 1:
        raise ValueError('bad output array: wrong shape')

    # calculate
    cdef:
        int limit = len(moments)
        int i, j
        complex *mom = <complex*>moments.data
        complex *o = <complex*>out.data
        double disc
    for i in prange(ndisc, nogil=True):
        o[i] = mom[0]
        for j in range(1, limit):
            o[i] += mom[j]*libc.math.cos(j*libc.math.M_PI*(2*i+1)/2/ndisc)*2
        disc = libc.math.cos(libc.math.M_PI*(2*i+1)/2/ndisc)
        o[i] /= libc.math.M_PI*libc.math.sqrt(1-disc*disc)

    # return
    return <object>out if do_return else None


#=======================#
# Fast cosine transform #
#=======================#


def fct(moments, ndisc=None, out=None):
    """Use discrete cosine transform to reconstruct target function. Use
    *ndisc* discretization steps. If *ndisc* is *None*, set ndisc=2*limit,
    where limit is the number of moments (truncation limit). If *out* is
    *None*, return the results, otherwise save them to the given array.

    This function uses :func:`scipy.fftpack.dct`.
    """
    raise NotImplementedError


#def _fct
#def _fct_real
#def _fct_real_omp
#def _fct_complex
#def _fct_complex_omp


#============================#
# Discrete fourier transform #
#============================#


def dft(moments):
    """Reconstruct target function using discrete fourier transform.
    """
    raise NotImplementedError


#def _dft
#def _dft_real
#def _dft_real_omp
#def _dft_complex
#def _dft_complex_omp


#========================#
# Fast fourier transform #
#========================#


def fft(moments):
    """Reconstruct target function using fast fourier transform algorithm.
    """
    raise NotImplementedError


#def _fft
#def _fft_real
#def _fft_real_omp
#def _fft_complex
#def _fft_complex_omp
