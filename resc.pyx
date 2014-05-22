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
"""Rescale matrix so that its spectrum fits into the interval [-1, 1]. Also,
rescale the results obtained by the kernel polynomial method back to the
original spectrum.

To do:
--> support other matrix formats besides CSR (e.g. dense matrices)
--> support expansions in more than one variable (multivariate functions)

Background: The kernel polynomial method (KPM) [1] uses a series expansion
technique using Chebychev polynomials. These polynomials are defined only
on the interval [-1, 1].

[1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)"""
__created__ = '2012-08-06'
__modified__ = '2012-10-09'
import numpy
import scipy.sparse
from cython.parallel import *
cimport numpy
import cofunc
import misc

#we need here:
# - function to rescale a given matrix
# - function to scale back any results (inverse rescale)
# - Lanczos wrapper that will be used if range is None


# also think about expansions in more than one variable
# e.g. with 2 variables we have: function defined on [-1, 1]x[-1, 1],
# parameters a, b, c, d needed


#======================#
# Rescaling the matrix #
#======================#


def rescale(mat, erange=None, params=None, omp=True, num_threads=None,
            eps=0., copy=True):
    """mat, (a, b) = rescale(mat, erange=None, params=None, omp=True,
                             num_threads=None, eps=0., copy=True)
    a, b = rescale(mat, erange=None, params=None, omp=True,
                   num_threads=None, eps=0., copy=False)

    Rescale the given matrix in-place. Specify either an energy range "erange"
    (2-tuple) or directly the scaling parameters "params" (2-tuple). If "omp"
    is True, use parallel for-loops (OpenMP). If "num_threads" is not None, set
    the number of threads. If the number of threads is smaller than 1,
    determine and use the number of processor cores.

    The matrix will be rescaled in-place like mat=(mat-b)/a, if a and b are the
    two scaling parameters. Instead of the parameters itself, an energy range
    can be specified. The eigenspectrum of the matrix should fit well into the
    given energy range. The scaling parameters are then calculated via
    a=(emax-emin)/(2-eps) and b=(emax+emin)/2, where (emin, emax) is the given
    energy range. An additional small number "eps" can be specified to get sure
    that the spectrum lies well inside the chosen energy range. For the Jackson
    kernel (see the submodule "kern"), eps=pi/limit is said to be an excellent
    choice [1], where limit is the number of moments (truncation limit) of the
    Chebychev expansion.

    This function delegates the actual work to one of the low-level functions,
    all beginning with "_rescale_", depending on the matrix type, data type and
    OpenMP-parallelization requirements. If no appropriate low-level function
    is found, then the plain Python implementation "_rescale" is used.

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006).
    """
    __created__ = '2012-08-13'
    __modified__ = '2012-10-09'
    # based on tb.kpm.KPMHDP.rescale developed from 2011-11-29 to 2012-07-16

    if copy:
        mat = mat.copy()

    # set number of threads
    if omp:
        misc.set_num_threads(num_threads)

    # either erange or params must be specified
    if (erange is None) + (params is None) != 1:
        raise ValueError('either "erange" or "params" must be specified')

    # get or calculate scaling parameters
    if erange is not None:
        a, b = erange2params(erange)
    else:
        try:
            a, b = params
        except:
            raise ValueError('"params" must be 2-tuple')

    # select and call appropriate low-level function
    #if mat.dtype is numpy.dtype(float):
        #func = _rescale_csr_real_omp if omp else _rescale_csr_real
    #elif mat.dtype is numpy.dtype(complex):
        #func = _rescale_csr_complex_omp if omp else _rescale_csr_complex
    #else:
    func = _rescale_sparse  # only this is used for now, already pretty fast
    mat = func(mat, a=a, b=b)

    # return
    return (mat, (a, b)) if copy else (a, b)


#===============================#
# Low-level rescaling functions #
#===============================#


def _rescale_sparse(mat, a=1., b=0.):
    """mat = _rescale(mat, a, b)

    Rescale given sparse matrix in-place. Use plain Python. Can handle any
    sparse matrix format and any data type, but is rather slow and only uses
    one processor core.
    """
    __created__ = '2012-08-15'
    __modified__ = '2012-09-11'

    # prepare
    if not isinstance(mat, scipy.sparse.base.spmatrix):
        raise TypeError('bad matrix format, expecting sparse matrix')
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('bad matrix shape, expecting square matrix')

    if b != 0:
        newdiag = mat.diagonal() - numpy.ones(mat.shape[0]) * b
        mat.setdiag(newdiag)
    mat = mat / a  # why is in-place-division not possible?
    return mat  # why is it not possible to change the matrix in-place?


def _rescale_csr_real(object mat, double a=1., double b=0.):
    """Rescale the given real square CSR sparse matrix "mat" in-place using the
    parameters "a" and "b", i.e. calculate mat=(mat-I*b)/a, where I is the
    identity matrix. Again, changes are made in-place, so be sure to work on a
    copy of the original matrix if necessary.

    Got some inspiration from
    https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h
    about how to access the diagonal entries of a CSR matrix."""
    __created__ = '2012-08-15'

    # prepare
    if type(mat) is not scipy.sparse.csr_matrix:
        raise TypeError('bad sparse matrix format, expecting CSR')
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('bad matrix shape, expecting square matrix')

    # calculate
    cdef:
        numpy.ndarray[double, mode="c"] data = mat.data
        numpy.ndarray[int, mode="c"] indices = mat.indices
        numpy.ndarray[int, mode="c"] indptr = mat.indptr
        double *dat = <double*>data.data
        int *ind = <int*>indices.data
        int *ptr = <int*>indptr.data
        int rank = mat.shape[0]
        int nnz = mat.nnz
        int rowstart, rowend, i, j
        int found = 0
    for i in range(rank):
        found = 0
        for j in range(ptr[i], ptr[i+1]):
            if ind[j] == i:
                found = 1
                dat[j] -= b
        if found == 0:
            # append new element mat[i, i] = -b
            # expensive!
            pass

    for j in range(nnz):
        dat[j] /= a
    raise NotImplementedError
    ### the problem still remains that the sparsity structure (the number of
    ### non-zeros) has to be changed. how to deal with that in C/Cython, even
    ### parallelized, is still unclear
    ### probably use numpy.append, numpy.insert, numpy.delete
    ### maybe use COO format


#===========================#
# Determine scaling factors #
#===========================#


def _opt2erange(string):
    """_opt2erange(string)

    Interprete a string (e.g. from a command line option) as a (an energy)
    range. Two comma-separated values (without whitespace) are accepted. If
    only one value a is given, return the range (-a, a).
    """
    __created__ = '2012-08-15'
    __modified__ = '2012-09-05'
    # based on tb.kpm.KPMHDP.get_erange, developed from 2011-11-29 to
    # 2012-08-15

    # interprete energy range by given string or value
    erange = misc.opt2mathlist(string, dtype=float)

    # determine energy range
    if len(erange) > 1:
        # get energy range
        emin, emax = erange
    elif len(erange) == 1:
        # assume symmetric energy range
        if erange[0] < 0:
            emin = erange[0]
            emax = -emin
        else:
            emax = erange[0]
            emin = -emax
    else:
        raise ValueError('bad energy range, expecting either one or two ' +
                         'values')

    # return energy range
    return emin, emax


def get_erange(mat, extra=0.):
    """emin, emax = get_erange(mat, extra=0.)

    If the (energy) spectrum of the (tight binding) matrix is completely
    unknown, this function can be used to estimate the highest and lowest
    (energy) eigenvalue by means of the Lanczos algorithm. (However, most of
    the time one should have a fairly good idea of the (energy) range already
    from the model that is used.)

    A small factor "extra" may be given, the returned (energy) range is
    enlarged by that factor to be sure that the whole (energy) spectrum of the
    given matrix is really contained in the returned range.
    """
    __created__ = '2012-08-15'
    __modified__ = '2012-08-22'
    # based on tb.kpm.KPMHDP.get_erange, developed from 2011-11-29 to
    # 2012-08-15

    # calculate highest and lowest eigenvalues using Lanczos algorithm
    some_eigvals = scipy.sparse.linalg.eigs(mat, k=12, which='LM',
                                            return_eigenvectors=False).real
    emin = min(some_eigvals)
    emax = max(some_eigvals)
    width = emax-emin

    # add an extra percentage to make sure no eigenvalue lies outside
    # (emin, emax)
    emin -= abs(extra*width)
    emax += abs(extra*width)

    return emin, emax


def erange2params(erange, eps=0.):
    """a, b = erange2params(erange, eps=0.)

    Calculate the scaling parameters "a" and "b" from the given energy range
    (2-tuple), like a=(emax-emin)/(2-eps) and b=(emax+emin)/2, where (emin,
    emax) is the given energy range "erange". An additional small number "eps"
    can be specified to make sure that the spectrum lies well inside the chosen
    energy range. For the Jackson kernel (see the submodule "kern"),
    eps=pi/limit is said to be an excellent choice [1], where limit is the
    number of moments (truncation limit) of the Chebychev expansion.

    [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006).
    """
    __created__ = '2012-08-22'
    emin, emax = _opt2erange(erange)
    a = (emax - emin) / (2 - eps)
    b = (emax + emin) / 2
    return a, b


#=====================================#
# Inversely rescaling the KPM results #
#=====================================#


def inverse_rescale(disc, params):
    """new_disc = inverse_rescale(disc, params)

    Scale the x-axis (discretization axis) of the quantity that has been
    calculated with the kernel polynomial method back to its original energy
    range. "params" is expecting a 2-tuple, containing the scaling parameters
    "a" and "b" that have been used before to rescale the matrix.

    The given discretization array "disc" is inversly rescaled like disc*a+b.

    """
    __created__ = '2012-08-22'
    try:
        a, b = params
    except:
        raise ValueError('param must be 2-tuple')
    return disc*a+b


def inverse_rescale_density(energ, dens, params):
    """new_energ, new_dens = inverse_rescale_density(energ, dens, params)

    Special case for densities that are calculated with the kernel polynomial
    method [1], where not only the energy axis (discretization) is scaled back
    to the original energy range of the matrix spectrum, but also the density
    itself is devided by the parameter "a", so that the density keeps
    its normalization."""
    __created__ = '2012-08-22'
    try:
        a, b = params
    except:
        raise ValueError('param must be 2-tuple')
    return inverse_rescale(energ, params), dens/a


def inverse_rescale_density_cofunc(obj, params):
    """new_cofunc = inverse_rescale_density_cofunc(cofunc, params)

    Same as "inverse_rescale_density", but accepts and returns a
    cofunc.coFunc object from the cofunc-module."""
    __created__ = '2012-08-22'
    return cofunc.coFunc(*inverse_rescale_density(obj.x, obj.y, params))


# http://comments.gmane.org/gmane.comp.python.cython.user/4723

# https://groups.google.com/forum/?fromgroups#!topic/cython-users/
# 7gKIBw8JqPQ[1-25]

# http://stackoverflow.com/questions/4641200/cython-inline-function-with-
# numpy-array-as-parameter

#cdef float** npy2c_float2d(np.ndarray[float, ndim=2, mode=c] a):
    #cdef float** a_c = malloc(a.shape[0] * sizeof(float*))
    #for k in range(a.shape[0]):
        #a_c[k] = &a[k, 0]
    #return a_c


# http://stackoverflow.com/a/9116735
#cdef np.ndarray[np.double_t, ndim=2, mode="c"] A_c
  #A_c = np.ascontiguousarray(A, dtype=np.double)
  #fc(&A_c[0,0])
