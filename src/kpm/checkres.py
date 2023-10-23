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
"""Helper functions to check results, e.g. get count or accuracy from GLDOS or
ALDOS
"""

import h5obj.tools
import numpy
from kpm import dummy


try:
    from comliner import Comliner
except ImportError:
    Comliner = dummy.Decorator


@Comliner()
def glcount(filename):
    """Get the attribute ``gldos.attrs.count`` of the dataset ``gldos`` from
    the given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/gldos').attrs.count
    except:
        return None


@Comliner()
def alcount(filename):
    """Get the attribute ``aldos.attrs.count`` of the dataset ``aldos`` from
    the given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/aldos').attrs.count
    except:
        return None


@Comliner()
def glacc(filename):
    """Get the attribute ``gldos.attrs.acc`` of the dataset ``gldos`` from the
    given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/gldos').attrs.acc
    except:
        return None


@Comliner()
def alacc(filename):
    """Get the attribute ``aldos.attrs.acc`` of the dataset ``aldos`` from the
    given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/aldos').attrs.acc
    except:
        return None


@Comliner()
def gllimit(filename):
    """Get the attribute ``gldos.attrs.limit`` of the dataset ``gldos`` from
    the given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/gldos').attrs.limit
    except:
        return None


@Comliner()
def allimit(filename):
    """Get the attribute ``aldos.attrs.limit`` of the dataset ``aldos`` from
    the given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/aldos').attrs.limit
    except:
        return None


@Comliner()
def acount(filename):
    """Get the attribute ``ados.attrs.count`` of the dataset ``ados`` from the
    given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/ados').attrs.count
    except:
        return None


@Comliner()
def aacc(filename):
    """Get the attribute ``ados.attrs.acc`` of the dataset ``ados`` from the
    given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/ados').attrs.acc
    except:
        return None


@Comliner()
def alimit(filename):
    """Get the attribute ``ados.attrs.limit`` of the dataset ``ados`` from the
    given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/ados').attrs.limit
    except:
        return None


@Comliner()
def lcount(filename):
    """Get the attribute ``ldos.attrs.count`` of the dataset ``ldos`` from the
    given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/ldos').attrs.count
    except:
        return None


@Comliner()
def lacc(filename):
    """Get the attribute ``ldos.attrs.acc`` of the dataset ``ldos`` from the
    given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/ldos').attrs.acc
    except:
        return None


@Comliner()
def llimit(filename):
    """Get the attribute ``ldos.attrs.limit`` of the dataset ``ldos`` from the
    given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/ldos').attrs.limit
    except:
        return None


@Comliner()
def dcount(filename):
    """Get the attribute ``dos.attrs.count`` of the dataset ``dos`` from the
    given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/dos').attrs.count
    except:
        return None


@Comliner()
def dacc(filename):
    """Get the attribute ``dos.attrs.acc`` of the dataset ``dos`` from the
    given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/dos').attrs.acc
    except:
        return None


@Comliner()
def dlimit(filename):
    """Get the attribute ``dos.attrs.limit`` of the dataset ``dos`` from the
    given HDF5 file.
    """
    try:
        return h5obj.tools.h5load(filename + '/dos').attrs.limit
    except:
        return None


@Comliner()
def glstderr0(filename):
    """Return the standard error of the GLDOS at zero energy.
    """
    try:
        gldos = h5obj.tools.h5load(filename+'/gldos')
        var = gldos.a2cf('var')
        var0 = var(0.)
        count = gldos.attrs.count
        stderr0 = numpy.sqrt(var0/(count-1))
        return stderr0
    except:
        return None


@Comliner()
def alstderr0(filename):
    """Return the standard error of the ALDOS at zero energy.
    """
    try:
        aldos = h5obj.tools.h5load(filename+'/aldos')
        var = aldos.a2cf('var')
        var0 = var(0.)
        count = aldos.attrs.count
        stderr0 = numpy.sqrt(var0/(count-1))
        return stderr0
    except:
        return None


@Comliner()
def astderr0(filename):
    """Return the standard error of the ADOS at zero energy.
    """
    try:
        ados = h5obj.tools.h5load(filename+'/ados')
        var = ados.a2cf('var')
        var0 = var(0.)
        count = ados.attrs.count
        stderr0 = numpy.sqrt(var0/(count-1))
        return stderr0
    except:
        return None


@Comliner(inmap=dict(gldos_list='$@/gldos', aldos_list='$@/aldos'))
def checksigma(gldos_list, aldos_list):
    """Investigate the fluctuation of the standard error of the geometric mean
    among independent calculations.
    """
    gvals = []
    avals = []
    gstderrs = []
    astderrs = []
    gammavals = []
    gammastds = []
    print
    print('GLDOS                ALDOS                Gamma')
    for gldos, aldos in zip(gldos_list, aldos_list):
        gldos0val = gldos(0.)
        aldos0val = aldos(0.)
        gvar = gldos.a2cf('var')
        avar = aldos.a2cf('var')
        gldos0var = gvar(0.)
        aldos0var = avar(0.)
        gcount = gldos.attrs.count
        acount = aldos.attrs.count
        gldos0stderr = numpy.sqrt(gldos0var/(gcount-1))
        aldos0stderr = numpy.sqrt(aldos0var/(acount-1))
        gvals.append(gldos0val)
        avals.append(aldos0val)
        gstderrs.append(gldos0stderr)
        astderrs.append(aldos0stderr)

        # calculate gamma
        gamma0val = gldos0val / aldos0val
        gamma0std = gldos0stderr/abs(aldos0val) \
            + aldos0stderr*abs(gldos0val/aldos0val**2)
        #gamma0var = gamma0std**2
        gammavals.append(gamma0val)
        gammastds.append(gamma0std)

        print('%.7f+-%.7f %.7f+-%.7f %.7f+-%.7f') \
            % (gldos0val, gldos0stderr, aldos0val, aldos0stderr,
               gamma0val, gamma0std)

    # compare with standard deviation
    gst = numpy.std(gvals)
    ast = numpy.std(avals)
    gammast = numpy.std(gammavals)
    print('STD: %.7f            %.7f            %.7f' % (gst, ast, gammast))


@Comliner()
def glval0(filename):
    """Return the GLDOS at zero energy.
    """
    try:
        gldos = h5obj.tools.h5load(filename+'/gldos')
        return gldos(0.)
    except:
        return None


@Comliner()
def alval0(filename):
    """Return the ALDOS at zero energy.
    """
    try:
        aldos = h5obj.tools.h5load(filename+'/aldos')
        return aldos(0.)
    except:
        return None


@Comliner()
def aval0(filename):
    """Return the ADOS at zero energy.
    """
    try:
        ados = h5obj.tools.h5load(filename+'/ados')
        return ados(0.)
    except:
        return None


