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
"""Miscellaneous function definitions, used in many places throughout the KPM
package.

This module is written in Cython, as some of the functions have to be called by
Cython functions in other modules.
"""

import numpy
import os
cimport openmp


def npc():
    """Return number of processor cores on this machine. Supported operating
    systems: Linux/Unix, MacOS, Windows.
    """
    # copied from tb.misc.npc from 2011-09-13
    # former tb.npc from 2011-02-10
    # former mytools.detectCPUs
    # based on code from http://www.boduch.ca/2009/06/python-cpus.html

    # Linux, Unix and Mac OS
    if hasattr(os, 'sysconf'):
        if 'SC_NPROCESSORS_ONLN' in os.sysconf_names:
            # Linux and Unix
            npc = os.sysconf('SC_NPROCESSORS_ONLN')
            if isinstance(npc, int) and npc > 0:
                return npc
            else:
                # Mac OS:
                return int(os.popen2('sysctl -n hw.ncpu')[1].read())

    # Windows
    if 'NUMBER_OF_PROCESSORS' in os.environ:
        npc = int(os.environ['NUMBER_OF_PROCESSORS'])
        if npc > 0:
            return npc

    # otherwise, return default value
    return 1


def symmetric_interval(x):
    """If a scalar *x* is given, return the tuple ``(-|x|, |x|)``.
    """
    if not hasattr(x, '__iter__'):
        x = [x]
    if len(x) == 2:
        return x[0], x[1]
    elif len(x) == 1:
        return -abs(x[0]), abs(x[0])
    else:
        raise ValueError('expecting scalar or iterable of length 1 or 2')


def set_num_threads(num_threads=None):
    """If *num_threads* is not *None*, set the number of OpenMP threads. If the
    number is smaller than 1, determine and use the number of processor
    cores.
    """
    if num_threads is not None:
        num_threads = int(num_threads)
        if num_threads < 1:
            num_threads = npc()  # use omp_get_max_threads instead?
        openmp.omp_set_num_threads(<unsigned int>num_threads)


def opt2mathlist(opt, dtype=int):
    """copy of tb.misc.opt2mathlist
    """
    if opt is None or opt == '':
        return []
    if isiterable(opt):
        return list(opt)

    # force string input
    opt = str(opt)

    result = []
    for r in opt.split(','):
        if r == '':
            if dtype == str:
                continue
            else:
                r = 0
        if ismath(r):
            r = evalmath(r, dtype=dtype)
        result.append(dtype(r))
    return result


def ismath(string):
    """Check if the given string is a mathematical expression (containing only
    mathematical operators like '+', '-', '*', or '/', and of course digits).
    Can be used to check if :func:`eval` is needed to evaluate an expression in
    a given string.

    Note: This function does not check if the numerical expression is actually
    valid. It just gives a hint if the given string should be passed to
    :func:`eval` or not.

    copy of tb.misc.ismath
    """
    if '+' in string or '*' in string or '/' in string:
        return True

    # special handling of minus sign
    if string.count('-') == 1 and string[0] == '-':
        return False
    if '-' in string:
        return True
    return False


def evalmath(value, dtype=float):
    """Cast value to the given data type *dtype*. If *value* is a string,
    assume that it contains a mathematical expression, and evaluate it with
    :func:`eval` before casting it to the specified type.

    The function could always use :func:`eval`, but this is assumed to be
    slower for values that do not have to be evaluated.

    copy of tb.misc.ismath
    """
    if isinstance(value, basestring) and ismath(value):
        #print(value, type(value), len(value))
        return dtype(eval(value.strip()))
    else:
        return dtype(value)


def isiterable(obj):
    """Check if the object *obj* is iterable. Return *True* for lists, tuples,
    dictionaries and numpy arrays (all objects that possess an __iter__
    method). Return *False* for scalars (*float*, *int*, etc.), strings, *bool*
    and *None*.

    copy of tb.misc
    """
    return not getattr(obj, '__iter__', False) is False


#def check_float1d(array):
  #if type(array) is not numpy.ndarray:
    #raise TypeError, 'bad type, expecting numpy.ndarray'
  #if array.dtype is not numpy.dtype(float):
    #raise ValueError, 'bad data type, expecting float (numpy.float64)'
  #if array.ndim != 1 or array.shape[0] < 1:
    #raise ValueError, 'bad data shape, expecting 1D array'


#def check_complex1d(array):
  #if type(array) is not numpy.ndarray:
    #raise TypeError, 'bad type, expecting numpy.ndarray'
  #if array.dtype is not numpy.dtype(complex):
    #raise ValueError, 'bad data type, expecting complex (numpy.complex128)'
  #if array.ndim != 1 or array.shape[0] < 1:
    #raise ValueError, 'bad data shape, expecting 1D array'


#def check_limit(limit):
  #if limit < 0:
    #raise ValueError, \
          #'bad truncation limit (%i), must be non-negative integer' % limit
