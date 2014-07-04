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
"""Create starting vectors for KPM algorithms."""
#
# To do:
# --> create real random-phase vectors using Gaussian distribution
# --> create real random-phase vectors using uniform distribution"""
__created__ = '2012-08-06'
__modified__ = '2014-01-14'
import numpy


def ind(length, index=0, dtype=float):
    """Return a specific basis state of the site-occupation basis (standard
    tight-binding basis), i.e. a vector of length *length* with the element at
    the given index *index* equal to 1 and the rest equal to 0, with datatype
    *dtype*."""
    # 2012-08-16
    length = int(length)
    index = int(index)
    if length < 1:
        raise ValueError('length must be positive integer')
    if index < 0 or index > length-1:
        raise ValueError('index out of bounds')
    vect = numpy.zeros(length, dtype=dtype)
    vect[index] = 1
    return vect


def randind(length, dtype=float):
    """Return a random basis state of the site-occupation basis (standard
    tight-binding basis), i.e. a vector of length *length* with exactly one
    random element equal to 1 and the rest equal to 0, with datatype
    *dtype*."""
    # 2012-08-16
    length = int(length)
    if length < 1:
        raise ValueError('length must be positive integer')
    vect = numpy.zeros(length, dtype=dtype)
    vect[numpy.random.randint(length)] = 1
    return vect


def randphase(length, dtype=complex):
    """Return a normalized random-phase vector."""
    # 2012-08-16 - 2014-01-14
    length = int(length)
    #phi = numpy.random.random_sample(size=(length,)*2)*2*numpy.pi
    phi = numpy.random.random_sample(size=length)*2*numpy.pi
    if dtype is complex:
        xi = numpy.exp(1j*phi)
        #print numpy.mean(xi), numpy.mean(xi**2), numpy.mean(xi**2)**2,
        #numpy.mean(xi**4)
        #print abs(xi), xi.shape
        #vect = xi[0]  # numpy.sum(xi, axis=-1)
        #vect = numpy.sum(xi, axis=-1)
        vect = xi
    elif dtype is float:
        raise NotImplementedError
        #return numpy.sqrt(2./3)*(1+numpy.cos(phi)) # why is this wrong?
        xi = numpy.cos(phi)
        #vect =
        #vect /= numpy.linalg.norm(vect)
    else:
        raise ValueError('datatype not supported')
    return vect


def randphase_backup(length, dtype=complex):
    """Return a normalized random-phase vector."""
    # 2012-08-16 - 2014-01-13
    length = int(length)
    phi = numpy.random.random_sample(size=length)*2*numpy.pi
    if dtype is complex:
        vect = numpy.exp(1j*phi)
    elif dtype is float:
        #return numpy.sqrt(2./3)*(1+numpy.cos(phi)) # why is this wrong?
        vect = numpy.cos(phi)
    else:
        raise ValueError('datatype not supported')
    vect /= numpy.linalg.norm(vect)
    return vect
