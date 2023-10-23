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
"""Define kernels. Introduce several functions that apply kernel damping
factors on some given Chebychev moments.

For an introduction into the kernels defined here, please refer to the
literature [1]. According to the literature [1], the Jackson kernel is best
suited for the calculation of spectral quantities like the density of states
(DOS) or the local density of states (LDOS).

References:

    - [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
"""

import cython
import numpy
cimport libc.math
cimport numpy
#cimport openmp
from cython.parallel import *
from kpm import misc


def select(string):
    """select(string)

    Select one of the kernels defined in this module by a given string.  For
    example, this could be an option string from a command line option.
    """
    string = str(string)
    if string and 'jackson'.startswith(string):
        return jackson
    elif string and 'lorentz'.startswith(string):
        return lorentz
    elif string and 'fejer'.startswith(string):
        return fejer
    elif string and 'lanczos'.startswith(string):
        return lanczos
    elif string and 'dirichlet'.startswith(string):
        return dirichlet
    else:
        raise ValueError('unknown kernel: %s' % string)


def all():
    """all()

    Return a list of all kernels defined in this module.
    """
    kernellist = [fejer, jackson, lorentz, lanczos, dirichlet]
    kernellist.sort()
    return kernellist


#================#
# Jackson kernel #
#================#


def jackson(moments, limit=None, omp=True, num_threads=None, out=None):
    """jackson(moments, limit=None, omp=True, num_threads=None, out=None)

    Apply the Jackson kernel to the given moments. If *limit* is *None* or
    smaller than 1, use the length of *moments* instead.

    This function only delegates the work to the corresponding low-level
    functions, all beginning with "_jackson_". If no optimal low-level function
    is found for the given datatype, the plain Python implementation "_jackson"
    is used.

    If *omp* is *True*, use available OpenMP-parallelized variants of the
    algorithms. If *num_threads* is not *None*, set the number of threads
    accordingly. If *num_threads* is smaller than 1, determine and use the
    number of processor cores instead.
    """
    if type(moments) is not numpy.ndarray:
        moments = numpy.ascontiguousarray(moments)
    if omp:
        misc.set_num_threads(num_threads)
    if moments.dtype is numpy.dtype(float):
        func = _jackson_real_omp if omp else _jackson_real
    elif moments.dtype is numpy.dtype(complex):
        func = _jackson_complex_omp if omp else _jackson_complex
    else:
        func = _jackson
    if limit is None:
        limit = 0
    return func(moments, limit=limit, out=out)


def _jackson(moments, limit=None, out=None):
    """_jackson(moments, limit=None, out=None)

    Apply the Jackson kernel to the given Chebychev moments using the given
    truncation limit *limit*. If *limit* is *None* or smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them.

    This is the pure Python variant of the function, using normal numpy arrays
    for the calculation.
    """
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if out.shape != moments.shape:
        raise ValueError('incompatible shapes')
    if limit is None or limit < 1:
        limit = len(moments)
    limit = int(limit)
    n = numpy.arange(len(moments))
    out = numpy.array(moments)*((limit+1-n)
                                * numpy.cos(numpy.pi * n / (limit + 1))
                                + numpy.sin(numpy.pi * n / (limit + 1))
                                / numpy.tan(numpy.pi / (limit + 1))) \
        / (limit + 1)
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _jackson_real(numpy.ndarray[double, mode="c"] moments,
                  int                             limit=0,
                  numpy.ndarray[double, mode="c"] out=None):
    """_jackson_real(ndarray[double, mode="c"] moments,
                    int                       limit=0,
                    ndarray[double, mode="c"] out=None)

    Apply the Jackson kernel to the given real Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments*. If *out* is not *None*, save the results in *out*
    instead of returning them.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        double *m = <double*>moments.data
        double *o = <double*>out.data
        int i
    for i in range(moments.shape[0]):
        o[i] = m[i]*((limit+1-i)
                     * libc.math.cos(libc.math.M_PI*i/(limit + 1))
                     + libc.math.sin(libc.math.M_PI*i/(limit + 1))
                     / libc.math.tan(libc.math.M_PI/(limit + 1)))/(limit + 1)

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _jackson_real_omp(numpy.ndarray[double, mode="c"] moments,
                      int                             limit=0,
                      numpy.ndarray[double, mode="c"] out=None):
    """_jackson_real_omp(ndarray[double, mode="c"] moments,
                        int                       limit=0,
                        ndarray[double, mode="c"] out=None)

    Apply the Jackson kernel to the given real Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them.

    This is the OpenMP variant of the function, using a parallel for-loop.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        double *m = <double*>moments.data
        double *o = <double*>out.data
        int i
    for i in prange(moments.shape[0], nogil=True):
        o[i] = m[i]*((limit+1-i)
                     * libc.math.cos(libc.math.M_PI*i/(limit+1))
                     + libc.math.sin(libc.math.M_PI*i/(limit+1))
                     / libc.math.tan(libc.math.M_PI/(limit+1)))/(limit+1)

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _jackson_complex(numpy.ndarray[complex, mode="c"] moments,
                     int                              limit=0,
                     numpy.ndarray[complex, mode="c"] out=None):
    """_jackson_complex(ndarray[complex, mode="c"] moments,
                    int                        limit=0,
                    ndarray[complex, mode="c"] out=None)

    Apply the Jackson kernel to the given complex Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        complex *m = <complex*>moments.data
        complex *o = <complex*>out.data
        int i
    for i in range(moments.shape[0]):
        o[i] = m[i]*((limit+1-i)
                     * libc.math.cos(libc.math.M_PI*i/(limit+1))
                     + libc.math.sin(libc.math.M_PI*i/(limit+1))
                     / libc.math.tan(libc.math.M_PI/(limit+1)))/(limit+1)

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _jackson_complex_omp(numpy.ndarray[complex, mode="c"] moments,
                         int                              limit=0,
                         numpy.ndarray[complex, mode="c"] out=None):
    """_jackson_complex_omp(ndarray[complex, mode="c"] moments,
                        int                        limit=0,
                        ndarray[complex, mode="c"] out=None)

    Apply the Jackson kernel to the given complex Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them.

    This is the OpenMP variant of the function, using a parallel for-loop.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        complex *m = <complex*>moments.data
        complex *o = <complex*>out.data
        int i
    for i in prange(moments.shape[0], nogil=True):
        o[i] = m[i]*((limit+1-i)
                     * libc.math.cos(libc.math.M_PI*i/(limit+1))
                     + libc.math.sin(libc.math.M_PI*i/(limit+1))
                     / libc.math.tan(libc.math.M_PI/(limit+1)))/(limit+1)

    # return
    return out if do_return else None


#==================#
# Dirichlet kernel #
#==================#


def dirichlet(moments, limit=None, omp=True, num_threads=None, out=None):
    """dirichlet(moments, limit=None, omp=True, num_threads=None, out=None)

    Apply the Dirichlet kernel to the given moments. This is the trivial kernel
    where the moments stay untouched (the kernel damping factors are all equal
    to 1). It is defined here just for the sake of completeness. All the
    arguments besides *moments* and *out* are just dummies.
    """
    if type(moments) is not numpy.ndarray:
        moments = numpy.array(moments)
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if out is not None:
        out = moments
    return moments if do_return else None


#==============#
# Fejer kernel #
#==============#


def fejer(moments, limit=None, omp=True, num_threads=None, out=None):
    """fejer(moments, limit=None, omp=True, num_threads=None, out=None)

    Apply the Fejer kernel to the given moments. If *limit* is *None* or
    smaller than 1, use the length of *moments*.

    This function only delegates the work to the corresponding low-level
    functions, all beginning with "_fejer_". If no optimal low-level function
    is found for the given datatype, the plain Python implementation "_fejer"
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
        func = _fejer_real_omp if omp else _fejer_real
    elif moments.dtype is numpy.dtype(complex):
        func = _fejer_complex_omp if omp else _fejer_complex
    else:
        func = _fejer
    if limit is None:
        limit = 0
    return func(moments, limit=limit, out=out)


def _fejer(moments, limit=None, out=None):
    """_fejer(moments, limit=None, out=None)

    Apply the Fejer kernel to the given Chebychev moments using the given
    truncation limit *limit*. If *limit* is *None* or smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them.

    This is the pure Python variant of the function, using normal numpy arrays
    for the calculation.
    """
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if out.shape != moments.shape:
        raise ValueError('incompatible shapes')
    if limit is None or limit < 1:
        limit = len(moments)
    limit = int(limit)
    n = numpy.arange(len(moments), dtype=float)
    out = numpy.array(moments)*(1-n/limit)
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _fejer_real(numpy.ndarray[double, mode="c"] moments,
                int                             limit=0,
                numpy.ndarray[double, mode="c"] out=None):
    """_fejer_real(ndarray[double, mode="c"] moments,
                int                       limit=0,
                ndarray[double, mode="c"] out=None)

    Apply the Fejer kernel to the given real Chebychev moments using the given
    truncation limit *limit*. If *limit* is smaller than 1, use the length of
    *moments* instead. If *out* is not *None*, save the results in *out*
    instead of returning them.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        double *m = <double*>moments.data
        double *o = <double*>out.data
        int i
    for i in range(moments.shape[0]):
        o[i] = m[i]*(1-1.*i/limit)

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _fejer_real_omp(numpy.ndarray[double, mode="c"] moments,
                    int                             limit=0,
                    numpy.ndarray[double, mode="c"] out=None):
    """_fejer_real_omp(ndarray[double, mode="c"] moments,
                    int                       limit=0,
                    ndarray[double, mode="c"] out=None)

    Apply the Fejer kernel to the given real Chebychev moments using the given
    truncation limit *limit*. If *limit* is smaller than 1, use the length of
    *moments* instead. If *out* is not *None*, save the results in *out*
    instead of returning them.

    This is the OpenMP variant of the function, using a parallel for-loop.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        double *m = <double*>moments.data
        double *o = <double*>out.data
        int i
    for i in prange(moments.shape[0], nogil=True):
        o[i] = m[i]*(1-1.*i/limit)

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _fejer_complex(numpy.ndarray[complex, mode="c"] moments,
                   int                              limit=0,
                   numpy.ndarray[complex, mode="c"] out=None):
    """_fejer_complex(ndarray[complex, mode="c"] moments,
                    int                        limit=0,
                    ndarray[complex, mode="c"] out=None)

    Apply the Fejer kernel to the given complex Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        complex *m = <complex*>moments.data
        complex *o = <complex*>out.data
        int i
    for i in range(moments.shape[0]):
        o[i] = m[i]*(1-1.*i/limit)

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _fejer_complex_omp(numpy.ndarray[complex, mode="c"] moments,
                       int                              limit=0,
                       numpy.ndarray[complex, mode="c"] out=None):
    """_fejer_complex_omp(ndarray[complex, mode="c"] moments,
                        int                        limit=0,
                        ndarray[complex, mode="c"] out=None)

    Apply the Fejer kernel to the given complex Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them.

    This is the OpenMP variant of the function, using a parallel for-loop.
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        complex *m = <complex*>moments.data
        complex *o = <complex*>out.data
        int i
    for i in prange(moments.shape[0], nogil=True):
        o[i] = m[i]*(1-1.*i/limit)

    # return
    return out if do_return else None


#================#
# Lorentz kernel #
#================#


def lorentz(moments, limit=None, param=4., omp=True, num_threads=None,
            out=None):
    """lorentz(moments, limit=None, param=4., omp=True, num_threads=None,
            out=None)

    Apply the Lorentz kernel to the given moments. If *limit* is *None* or
    smaller than 1, use the length of *moments* instead. *param* is a free
    parameter, but it is said to be optimal between 3.0 and 5.0 [1].

    This function only delegates the work to the corresponding low-level
    functions, all beginning with "_lorentz_". If no optimal low-level function
    is found for the given datatype, the plain Python implementation "_lorentz"
    is used.

    If *omp* is *True*, use available OpenMP-parallelized variants of the
    algorithms. If *num_threads* is not *None*, set the number of threads
    accordingly. If *num_threads* is smaller than 1, determine and use the
    number of processor cores.

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """
    param = float(param)
    if type(moments) is not numpy.ndarray:
        moments = numpy.ascontiguousarray(moments)
    if omp:
        misc.set_num_threads(num_threads)
    if moments.dtype is numpy.dtype(float):
        func = _lorentz_real_omp if omp else _lorentz_real
    elif moments.dtype is numpy.dtype(complex):
        func = _lorentz_complex_omp if omp else _lorentz_complex
    else:
        func = _lorentz
    if limit is None:
        limit = 0
    return func(moments, limit=limit, param=param, out=out)


def _lorentz(moments, limit=None, param=4., out=None):
    """_lorentz(moments, limit=None, param=4., out=None)

    Apply the Lorentz kernel to the given Chebychev moments using the given
    truncation limit *limit*. If *limit* is *None* or smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them. *param* is a free parameter, but it is
    said to be optimal between 3.0 and 5.0 [1].

    This is the pure Python variant of the function, using normal numpy arrays
    for the calculation.

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if out.shape != moments.shape:
        raise ValueError('incompatible shapes')
    if limit is None or limit < 1:
        limit = len(moments)
    limit = int(limit)
    n = numpy.arange(len(moments), dtype=float)
    out = numpy.array(moments)*numpy.sinh(param*(1-n/limit)) \
        / numpy.sinh(param)
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _lorentz_real(numpy.ndarray[double, mode="c"] moments,
                  int                             limit=0,
                  double                          param=4.,
                  numpy.ndarray[double, mode="c"] out=None):
    """_lorentz_real(ndarray[double, mode="c"] moments,
                    int                       limit=0,
                    double                    param=4.,
                    ndarray[double, mode="c"] out=None)

    Apply the Lorentz kernel to the given real Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them. *param* is a free parameter, but it is
    said to be optimal between 3.0 and 5.0 [1].

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        double *m = <double*>moments.data
        double *o = <double*>out.data
        int i
    for i in range(moments.shape[0]):
        o[i] = m[i]*libc.math.sinh(param*(1-1.*i/limit)) \
            / libc.math.sinh(param)

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _lorentz_real_omp(numpy.ndarray[double, mode="c"] moments,
                      int                             limit=0,
                      double                          param=4.,
                      numpy.ndarray[double, mode="c"] out=None):
    """_lorentz_real_omp(ndarray[double, mode="c"] moments,
                        int                       limit=0,
                        double                    param=4.,
                        ndarray[double, mode="c"] out=None)

    Apply the Lorentz kernel to the given real Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them. *param* is a free parameter, but it is
    said to be optimal between 3.0 and 5.0 [1].

    This is the OpenMP variant of the function, using a parallel for-loop.

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        double *m = <double*>moments.data
        double *o = <double*>out.data
        int i
    for i in prange(moments.shape[0], nogil=True):
        o[i] = m[i]*libc.math.sinh(param*(1-1.*i/limit)) \
            / libc.math.sinh(param)

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _lorentz_complex(numpy.ndarray[complex, mode="c"] moments,
                     int                              limit=0,
                     double                           param=4.,
                     numpy.ndarray[complex, mode="c"] out=None):
    """_lorentz_complex(ndarray[complex, mode="c"] moments,
                    int                        limit=0,
                    double                     param=4.,
                    ndarray[complex, mode="c"] out=None)

    Apply the Lorentz kernel to the given complex Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them. *param* is a free parameter, but it is
    said to be optimal between 3.0 and 5.0 [1].

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        complex *m = <complex*>moments.data
        complex *o = <complex*>out.data
        int i
    for i in range(moments.shape[0]):
        o[i] = m[i]*libc.math.sinh(param*(1-1.*i/limit)) \
            / libc.math.sinh(param)

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _lorentz_complex_omp(numpy.ndarray[complex, mode="c"] moments,
                         int                              limit=0,
                         double                           param=4.,
                         numpy.ndarray[complex, mode="c"] out=None):
    """_lorentz_complex_omp(ndarray[complex, mode="c"] moments,
                        int                        limit=0,
                        double                     param=4.,
                        ndarray[complex, mode="c"] out=None)

    Apply the Lorentz kernel to the given complex Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them. *param* is a free parameter, but it is
    said to be optimal between 3.0 and 5.0 [1].

    This is the OpenMP variant of the function, using a parallel for-loop.

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        complex *m = <complex*>moments.data
        complex *o = <complex*>out.data
        int i
    for i in prange(moments.shape[0], nogil=True):
        o[i] = m[i]*libc.math.sinh(param*(1-1.*i/limit)) \
            / libc.math.sinh(param)

    # return
    return out if do_return else None


#================#
# Lanczos kernel #
#================#


def lanczos(moments, limit=None, param=3, omp=True, num_threads=None,
            out=None):
    """lanczos(moments, limit=None, param=3, omp=True, num_threads=None,
            out=None)

    Apply the Lanczos kernel to the given moments. If *limit* is *None* or
    smaller than 1, use the length of *moments* instead. *param* is a free
    integer parameter. It is said that this kernel comes close to the Jackson
    kernel for param=3 [1].

    This function only delegates the work to the corresponding low-level
    functions, all beginning with "_lanczos_". If no optimal low-level function
    is found for the given datatype, the plain Python implementation "_lanczos"
    is used.

    If *omp* is *True*, use available OpenMP-parallelized variants of the
    algorithms. If *num_threads* is not None, set the number of threads
    accordingly. If *num_threads* is smaller than 1, determine and use the
    number of processor cores.

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """
    param = int(param)
    if type(moments) is not numpy.ndarray:
        moments = numpy.ascontiguousarray(moments)
    if omp:
        misc.set_num_threads(num_threads)
    if moments.dtype is numpy.dtype(float):
        func = _lanczos_real_omp if omp else _lanczos_real
    elif moments.dtype is numpy.dtype(complex):
        func = _lanczos_complex_omp if omp else _lanczos_complex
    else:
        func = _lanczos
    if limit is None:
        limit = 0
    return func(moments, limit=limit, param=param, out=out)


def _lanczos(moments, limit=None, param=3, out=None):
    """_lanczos(moments, limit=None, param=3, out=None)

    Apply the Lanczos kernel to the given Chebychev moments using the given
    truncation limit *limit*. If *limit* is None or smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in *out*
    instead of returning them. *param* is a free integer parameter. It is said
    that this kernel comes close to the Jackson kernel for param=3 [1].

    This is the pure Python variant of the function, using normal numpy arrays
    for the calculation.

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if out.shape != moments.shape:
        raise ValueError('incompatible shapes')
    if limit is None or limit < 1:
        limit = len(moments)
    limit = int(limit)
    n = numpy.arange(len(moments), dtype=float)
    g = numpy.empty_like(n)
    g[0] = 1.
    g[1:] = (numpy.sin(numpy.pi*n[1:]/limit)/numpy.pi/n[1:]*limit)**param
    out = numpy.array(moments)*g
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _lanczos_real(numpy.ndarray[double, mode="c"] moments,
                  int                             limit=0,
                  int                             param=3,
                  numpy.ndarray[double, mode="c"] out=None):
    """_lanczos_real(ndarray[double, mode="c"] moments,
                    int                       limit=0,
                    int                       param=3,
                    ndarray[double, mode="c"] out=None)

    Apply the Lanczos kernel to the given real Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them. *param* is a free integer parameter. It is
    said that this kernel comes close to the Jackson kernel for param=3 [1].

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        double *m = <double*>moments.data
        double *o = <double*>out.data
        int i
    o[0] = m[0]
    for i in range(1, moments.shape[0]):
        o[i] = m[i]*(libc.math.sin(i*libc.math.M_PI/limit)
                     / i/libc.math.M_PI*limit)**param

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _lanczos_real_omp(numpy.ndarray[double, mode="c"] moments,
                      int                             limit=0,
                      int                             param=3,
                      numpy.ndarray[double, mode="c"] out=None):
    """_lanczos_real_omp(ndarray[double, mode="c"] moments,
                        int                       limit=0,
                        int                       param=3,
                        ndarray[double, mode="c"] out=None)

    Apply the Lanczos kernel to the given real Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them. *param* is a free integer parameter. It is
    said that this kernel comes close to the Jackson kernel for param=3 [1].

    This is the OpenMP variant of the function, using a parallel for-loop.

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        double *m = <double*>moments.data
        double *o = <double*>out.data
        int i
    o[0] = m[0]
    for i in prange(1, moments.shape[0], nogil=True):
        o[i] = m[i]*(libc.math.sin(i*libc.math.M_PI/limit)
                     / i/libc.math.M_PI*limit)**param

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _lanczos_complex(numpy.ndarray[complex, mode="c"] moments,
                     int                              limit=0,
                     int                              param=3,
                     numpy.ndarray[complex, mode="c"] out=None):
    """_lanczos_complex(ndarray[complex, mode="c"] moments,
                    int                        limit=0,
                    int                        param=3,
                    ndarray[complex, mode="c"] out=None)

    Apply the Lanczos kernel to the given complex Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them. *param* is a free integer parameter. It is
    said that this kernel comes close to the Jackson kernel for param=3 [1].

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        complex *m = <complex*>moments.data
        complex *o = <complex*>out.data
        int i
    o[0] = m[0]
    for i in range(1, moments.shape[0]):
        o[i] = m[i]*(libc.math.sin(i*libc.math.M_PI/limit)
                     / i/libc.math.M_PI*limit)**param

    # return
    return out if do_return else None


@cython.cdivision(True)
@cython.boundscheck(False)
def _lanczos_complex_omp(numpy.ndarray[complex, mode="c"] moments,
                         int                              limit=0,
                         int                              param=3,
                         numpy.ndarray[complex, mode="c"] out=None):
    """_lanczos_complex_omp(ndarray[complex, mode="c"] moments,
                        int                        limit=0,
                        int                        param=3,
                        ndarray[complex, mode="c"] out=None)

    Apply the Lanczos kernel to the given complex Chebychev moments using the
    given truncation limit *limit*. If *limit* is smaller than 1, use the
    length of *moments* instead. If *out* is not *None*, save the results in
    *out* instead of returning them. *param* is a free integer parameter. It is
    said that this kernel comes close to the Jackson kernel for param=3 [1].

    This is the OpenMP variant of the function, using a parallel for-loop.

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    """

    # prepare
    if not moments.flags['C_CONTIGUOUS']:
        moments = numpy.ascontiguousarray(moments)
    do_return = out is None
    if out is None:
        out = numpy.empty_like(moments)
    if not out.flags['C_CONTIGUOUS']:
        out = numpy.ascontiguousarray(out)
    if out.shape[0] != moments.shape[0]:
        raise ValueError('bad output array: wrong shape')
    if limit < 1:
        limit = len(moments)

    # calculate
    cdef:
        complex *m = <complex*>moments.data
        complex *o = <complex*>out.data
        int i
    o[0] = m[0]
    for i in prange(1, moments.shape[0], nogil=True):
        o[i] = m[i]*(libc.math.sin(i*libc.math.M_PI/limit)
                     / i/libc.math.M_PI*limit)**param

    # return
    return out if do_return else None
