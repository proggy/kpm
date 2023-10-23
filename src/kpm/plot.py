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
"""KPM-related plot functions.
"""

import matplotlib.pyplot as plt
from kpm import dummy
import numpy

try:
    from comliner import Comliner, items_of
except ImportError:
    Comliner = dummy.Decorator


# common comliner configuration for all comliners defined in this module
optdoc = dict(err='plot errorbars')
outmap = {0: None}
#shortopts = dict()
#longopts = dict()
#opttypes = dict()


@Comliner(inmap=dict(ldos='$0/ldos', label='$0'), outmap=outmap,
      optdoc=optdoc, last=plt.show)
def pldos(ldos, err=False, fmt='-', label=None):
    """Plot LDOS (local density of states). Expect :class:`cofunc.coFunc`
    object.
    """
    yerr = 2*numpy.sqrt(ldos.attrs.var/ldos.attrs.count) if err else None
    label = label+'/ldos' if label else None
    ax = plt.errorbar(ldos.x, ldos.y, yerr=yerr, fmt=fmt, label=label)
    plt.xlabel('energy')
    plt.ylabel('LDOS')
    plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.pldos')
    return ax


@Comliner(inmap=dict(dos='$0/dos', label='$0'), outmap=outmap,
      optdoc=optdoc, last=plt.show)
def pdos(dos, err=False, fmt='-', label=None):
    """Plot DOS (density of states). Expect :class:`cofunc.coFunc` object.
    """
    yerr = 2*numpy.sqrt(dos.attrs.var/dos.attrs.count) if err else None
    label = label+'/dos' if label else None
    ax = plt.errorbar(dos.x, dos.y, yerr=yerr, fmt=fmt, label=label)
    plt.xlabel('energy')
    plt.ylabel('DOS')
    plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.pdos')
    return ax


@Comliner(inmap=dict(ados='$0/ados', label='$0'), outmap=outmap,
      optdoc=optdoc, last=plt.show)
def pados(ados, err=False, fmt='-', label=None):
    """Plot ADOS (density of states). Expect :class:`cofunc.coFunc` object.
    """
    yerr = 2*numpy.sqrt(ados.attrs.var/ados.attrs.count) if err else None
    label = label+'/ados' if label else None
    ax = plt.errorbar(ados.x, ados.y, yerr=yerr, fmt=fmt, label=label)
    plt.xlabel('energy')
    plt.ylabel('ADOS')
    plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.pados')
    return ax


@Comliner(inmap=dict(aldos='$0/aldos', label='$0'), outmap=outmap,
      optdoc=optdoc, last=lambda: plt.show())
def paldos(aldos, err=False, fmt='-', label=None):
    """Plot ALDOS (arithmetic mean of the local density of states). Expect
    :class:`cofunc.coFunc` object. If *err* is *True*, calculate the standard
    error (using :class:`cofunc.coFunc` attributes *var* and *count*) and plot
    errorbars.
    """
    yerr = 2*numpy.sqrt(aldos.attrs.var/aldos.attrs.count) if err else None
    label = label if label else None
    ax = plt.errorbar(aldos.x, aldos.y, yerr=yerr, fmt=fmt, label=label)
    plt.xlabel('energy')
    plt.ylabel('ALDOS')
    plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.paldos')
    return ax


@Comliner(inmap=dict(aldoslist='$@/aldos', disvals='$@/param'), outmap=outmap,
      optdoc=optdoc, shortopts=dict(err='b'), opttypes=dict(ymin=float),
      preproc=dict(disvals=items_of('scale'),
                   energs=lambda x: [float(i) for i in x.split(',')]
                   if isinstance(x, basestring) else x))
def paldos_energ(disvals, aldoslist, energs=[0.], err=False, ymin=None,
                 fmt='.:'):

    """Plot ALDOS at the given energies for different disorder. If *err* is
    *True*, calculate the standard error (using :class:`cofunc.coFunc`
    attributes *var* and *count*) and plot errorbars (95 % confidence
    intervals).
    """
    for energ in energs:
        aldosvals = []
        aldoserrs = []
        for aldos in aldoslist:
            aldosvals.append(aldos(energ))
            varval = aldos.a2cf('var')(energ)
            errval = numpy.sqrt(varval / (aldos.attrs.count - 1))
            aldoserrs.append(2*errval)
        if not err:
            aldoserrs = None
        ax = plt.errorbar(disvals, aldosvals, yerr=aldoserrs, fmt=fmt,
                          label='E=%.2f' % energ)

    if len(energs) > 1:
        plt.legend()
    plt.axis([None, None, ymin, None])
    plt.xlabel('disorder')
    plt.ylabel('ALDOS')
    plt.gcf().canvas.set_window_title(__name__+'.paldos-energ')
    plt.show()
    return ax


def last():
    plt.xlabel('energy')
    plt.ylabel('GLDOS')
    plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.pgldos')
    plt.show()


@Comliner(inmap=dict(gldos='$0/gldos', label='$0'), outmap=outmap,
      optdoc=optdoc, last=last)
def pgldos(gldos, err=False, fmt='-', color=None, label=None):
    """Plot GLDOS (geometric mean of the local density of states, or typical
    density of states). Expect :class:`cofunc.coFunc` object. If *err* is
    *True*, calculate the standard error (using :class:`cofunc.coFunc`
    attributes *var* and *count*) and plot errorbars.
    """
    color = color or next(plt.gca()._get_lines.color_cycle)
    yerr = 2*numpy.sqrt(gldos.attrs.var/gldos.attrs.count) if err else None
    label = label if label else None
    ax = plt.errorbar(gldos.x, gldos.y, yerr=yerr, fmt=fmt, color=color,
                      label=label)
    return ax


def last():
    plt.xlabel('energy')
    plt.ylabel('ADOS and GLDOS')
    plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.pagldos')
    plt.show()


@Comliner(inmap=dict(ados='$0/ados', gldos='$0/gldos', label='$0'),
      outmap=outmap, optdoc=optdoc, last=last)
def pagldos(ados, gldos, err=False, color=None, label=None):
    """Plot ADOS (average density of states) and GLDOS (geometric mean of the
    local density of states). Expect two :class:`cofunc.coFunc` objects.  If
    *err* is *True*, calculate the standard error (using :class:`cofunc.coFunc`
    attributes *var* and *count*) and plot errorbars.
    """
    color = color or next(plt.gca()._get_lines.color_cycle)
    aerr = 1.96*numpy.sqrt(ados.attrs.var/ados.attrs.count) if err else None
    gerr = 1.96*numpy.sqrt(gldos.attrs.var/gldos.attrs.count) if err else None
    alabel = label + '/ados' if label else None
    glabel = label + '/gldos' if label else None
    plt.errorbar(ados.x, ados.y, yerr=aerr, fmt='-', color=color, label=alabel)
    ax = plt.errorbar(gldos.x, gldos.y, yerr=gerr, fmt=':', color=color,
                      label=glabel)
    return ax


def last():
    plt.xlabel('energy')
    plt.ylabel('ALDOS and GLDOS')
    plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.pgaldos')
    plt.show()


@Comliner(inmap=dict(aldos='$0/aldos', gldos='$0/gldos', label='$0'),
      outmap=outmap, optdoc=optdoc, last=last)
def pgaldos(aldos, gldos, err=False, color=None, label=None):
    """Plot ALDOS (arithmetic mean of the local density of states) and GLDOS
    (geometric mean of the local density of states). Expect two
    :class:`cofunc.coFunc` objects.  If *err* is *True*, calculate the standard
    error (using :class:`cofunc.coFunc` attributes *var* and *count*) and plot
    errorbars.
    """
    color = color or next(plt.gca()._get_lines.color_cycle)
    aerr = 2*numpy.sqrt(aldos.attrs.var/aldos.attrs.count) if err else None
    gerr = 2*numpy.sqrt(gldos.attrs.var/gldos.attrs.count) if err else None
    alabel = label + '/aldos' if label else None
    glabel = label + '/gldos' if label else None
    plt.errorbar(aldos.x, aldos.y, yerr=aerr, fmt='-', color=color,
                 label=alabel)
    ax = plt.errorbar(gldos.x, gldos.y, yerr=gerr, fmt=':', color=color,
                      label=glabel)
    return ax


def last():
    plt.xlabel('$\Gamma$')
    plt.ylabel('count')
    plt.title('Histogram')
    #plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.pchecksigma')
    plt.show()


@Comliner(inmap=dict(gldos_list='$@/gldos', aldos_list='$@/aldos'), outmap=outmap,
      optdoc=optdoc, last=last)
def pchecksigma(gldos_list, aldos_list, bins=20, color=None, label=None):
    """Investigate the fluctuation of the standard error of the geometric mean
    among independent calculations in form of a histogram.
    """
    color = color or next(plt.gca()._get_lines.color_cycle)
    gvals = []
    avals = []
    gstderrs = []
    astderrs = []
    gammavals = []
    gammastds = []
    gammastds2 = []
    gammastds3 = []
    print
    print 'GLDOS                 ALDOS                 Gamma'
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

        # propagate the standard deviation another way
        gamma0std2 = numpy.sqrt(gldos0stderr**2/aldos0val**2
            + aldos0stderr**2*gldos0val**2/aldos0val**4)
        gammastds2.append(gamma0std2)

        # yet another way, assuming the correlation coefficient between GLDOS
        # and ALDOS is zero
        gamma0std3 = gamma0val*numpy.sqrt(gldos0stderr**2/gldos0val**2
            + aldos0stderr**2/aldos0val**2)
        gammastds3.append(gamma0std3)

        print '%.7f+-%.7f  %.7f+-%.7f  %.7f+-%.7f  +-%.7f  +-%.7f' \
            % (gldos0val, gldos0stderr, aldos0val, aldos0stderr,
               gamma0val, gamma0std, gamma0std2, gamma0std3)

    # compare with standard deviation
    gst = numpy.std(gvals)
    ast = numpy.std(avals)
    gammast = numpy.std(gammavals)
    print 'STD: %.7f             %.7f             %.7f' % (gst, ast, gammast)

    # plot histogram
    ax = plt.hist(gammavals, bins=bins, color=color, label=label)
    return ax


def repeat_items(sequence, n=2):
    """Return new sequence where each item of the old *sequence* repeats the
    given number *n* of times.
    """
    new = []
    for item in sequence:
        new += [item] * n
    return new
