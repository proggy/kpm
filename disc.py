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
"""Discretize target function. Find a suitable discretization for the
independent variables of the target function. For example, in the case of the
(local) density of states, this will be the energy axis, represented by a
1D-array.

Note that this module works with the rescaled axes, so the output is always
some discretization of the interval [-1, 1]. The target quantity still has to
be scaled back to the original matrix spectrum, together with the discretized
x-axis."""
#
# Todo:
# --> Cython/OpenMP versions? Not really necessary...
#
# Ideas:
# --> support to only get part of the spectrum (some interval within [-1, 1])
#
__created__ = '2012-08-13'
__modified__ = '2013-06-25'
import numpy


def uniform(ndisc, eps=1e-5):
    """Return a uniform discretization of the interval [-1, 1]. If *ndisc* is
    of type *int*, use *ndisc* discretization steps. If *ndisc* is of type
    *float*, use it as the stepwidth. The number of steps is then determined by
    the stepwidth and the boundaries (-1, 1).

    Note: The returned list of numbers will always be symmetric to 0. An odd
    number of steps will always include 0. Likewise, a given stepwidth will
    always result in an odd number of values including 0."""
    # 2012-08-18 - 2013-07-21
    if isinstance(ndisc, basestring):
        if '.' in ndisc:
            ndisc = float(ndisc)
        else:
            ndisc = int(ndisc)
    if isinstance(ndisc, float):
        stepwidth = ndisc
        positive_half = numpy.arange(abs(stepwidth), 1-eps, abs(stepwidth))
        return numpy.r_['-1', -positive_half[::-1], [0], positive_half]
    else:
        return numpy.linspace(-1+eps, 1-eps, int(ndisc))


def cosine(ndisc):
    """Return cosine-like discretization of the interval [-1, 1], using *ndisc*
    discretization steps. This form of discretization is needed if the discrete
    cosine transform (dct) is being used for reconstructing the target function
    (see rcstr-module). The default for *ndisc* should be 2*limit, where
    limit is the number of moments (truncation limit).

    This is the pure Python version of this function, using normal Numpy
    functions."""
    # 2012-08-18 - 2013-06-20
    ndisc = int(ndisc)
    return numpy.cos(numpy.pi*(2*numpy.arange(ndisc, dtype=float)+1)/2/ndisc)
