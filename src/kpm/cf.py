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
"""This submodule implements special wrappers for certain KPM functions that
use :class:`cofunc.coFunc` objects for input and output instead of simple numpy
arrays."""

import cofunc
import kpm
from kpm import dummy

try:
    from comliner import Comliner
except ImportError:
    Comliner = dummy.Decorator


@Comliner(inmap=dict(mat='$0/scell'), outmap={0: '$0/ldos'},
      preproc=dict(mat=lambda scell: scell.tbmat(format='csr')),
      shortopts=kpm.shortopts, longopts=kpm.longopts,
      opttypes=kpm.opttypes, optdoc=kpm.optdoc)
def ldos(mat, state=0, limit=100, erange=10, enum=None, estep=None,
         kernel='jackson', rcstr_method='std', omp=False, num_threads=None,
         rescaled=False, stateclass=''):
    """Wrapper for :func:`kpm.ldos` that returns a :class:`cofunc.coFunc`
    object.
    """
    energ, dens \
        = kpm.ldos(mat, state=state, limit=limit, erange=erange, enum=enum,
                   estep=estep, kernel=kernel, rcstr_method=rcstr_method,
                   omp=omp, num_threads=num_threads, rescaled=rescaled,
                   stateclass=stateclass)
    ldos = cofunc.coFunc(energ, dens)
    ldos.attrs.update(state=state, limit=limit, erange=erange, enum=enum,
                      estep=estep, kernel=kernel, rcstr_method=rcstr_method,
                      omp=omp, num_threads=num_threads, rescaled=rescaled,
                      stateclass=stateclass)
    return ldos


@Comliner(inmap=dict(scell='$0/scell', init='$0/aldos'), outmap={0: '$0/aldos'},
      shortopts=kpm.shortopts, longopts=kpm.longopts,
      opttypes=kpm.opttypes, optdoc=kpm.optdoc, overwrite=True)
def aldos(scell, erange=10, enum=None, estep=None, count=None, tol=None,
          smooth=10, limit=100, kernel='jackson', rcstr_method='std',
          stateclass=None, spr=1, omp=False, num_threads=0,
          until=None, verbose=False, init=None):
    """Wrapper for :func:`kpm.aldos` that uses :class:`cofunc.coFunc` objects
    for input and output.
    """
    #init_energ = init.x if init else None
    init_dens = init.y if init else None
    init_var = init.attrs.var if init else None
    init_count = init.attrs.count if init else None

    energ, dens, var, count, acc \
        = kpm.aldos(scell, erange=erange, enum=enum, estep=estep, count=count,
                    tol=tol, smooth=smooth,
                    limit=limit, kernel=kernel, rcstr_method=rcstr_method,
                    stateclass=stateclass, spr=spr,
                    omp=omp, num_threads=num_threads,
                    until=until, verbose=verbose,
                    init_dens=init_dens,  # init_energ=init_energ
                    init_count=init_count, init_var=init_var)

    if energ is None and init:
        energ = init.x
    aldos = cofunc.coFunc(energ, dens)
    aldos.attrs.update(erange=erange, enum=enum, estep=estep,
                       tol=tol, smooth=smooth,
                       limit=limit, kernel=kernel, rcstr_method=rcstr_method,
                       stateclass=stateclass, spr=spr,
                       omp=omp, num_threads=num_threads,
                       until=until, verbose=verbose, var=var, count=count,
                       acc=acc)
    return aldos


@Comliner(inmap=dict(scell='$0/scell', init='$0/gldos'), outmap={0: '$0/gldos'},
      shortopts=kpm.shortopts, longopts=kpm.longopts,
      opttypes=kpm.opttypes, optdoc=kpm.optdoc, overwrite=True)
def gldos(scell, erange=10, enum=None, estep=None, count=None, tol=None,
          smooth=10, limit=100, kernel='jackson', rcstr_method='std',
          stateclass=None, spr=1, omp=False, num_threads=0,
          until=None, verbose=False, init=None):
    """Wrapper for :func:`kpm.gldos` that uses :class:`cofunc.coFunc` objects
    for input and output.
    """
    #init_energ = init.x if init else None
    init_dens = init.y if init else None
    init_var = init.attrs.var if init else None
    init_count = init.attrs.count if init else None

    energ, dens, var, count, acc \
        = kpm.gldos(scell, erange=erange, enum=enum, estep=estep, count=count,
                    tol=tol, smooth=smooth,
                    limit=limit, kernel=kernel, rcstr_method=rcstr_method,
                    stateclass=stateclass, spr=spr,
                    omp=omp, num_threads=num_threads,
                    until=until, verbose=verbose,
                    init_dens=init_dens,  # init_energ=init_energ
                    init_count=init_count, init_var=init_var)

    if energ is None and init:
        energ = init.x
    gldos = cofunc.coFunc(energ, dens)
    gldos.attrs.update(erange=erange, enum=enum, estep=estep,
                       tol=tol, smooth=smooth,
                       limit=limit, kernel=kernel, rcstr_method=rcstr_method,
                       stateclass=stateclass, spr=spr,
                       omp=omp, num_threads=num_threads,
                       until=until, verbose=verbose, var=var, count=count,
                       acc=acc)
    return gldos


@Comliner(inmap=dict(scell='$0/scell', ainit='$0/aldos', ginit='$0/gldos'),
      outmap={0: '$0/aldos', 1: '$0/gldos'},
      shortopts=kpm.shortopts, longopts=kpm.longopts,
      opttypes=kpm.opttypes, optdoc=kpm.optdoc, overwrite=True)
def galdos(scell, erange=10, enum=None, estep=None, count=None, tol=None,
           smooth=10, limit=100, kernel='jackson', rcstr_method='std',
           stateclass=None, spr=1, omp=False, num_threads=0,
           until=None, verbose=False, ainit=None, ginit=None):
    """Wrapper for :func:`kpm.galdos` that uses :class:`cofunc.coFunc` objects
    for input and output.
    """
    #init_aenerg = ainit.x if ainit else None
    init_adens = ainit.y if ainit else None
    init_avar = ainit.attrs.var if ainit else None
    init_acount = ainit.attrs.count if ainit else None
    #init_generg = ginit.x if ginit else None
    init_gdens = ginit.y if ginit else None
    init_gvar = ginit.attrs.var if ginit else None
    init_gcount = ginit.attrs.count if ginit else None

    aenerg, adens, avar, acount, aacc, generg, gdens, gvar, gcount, gacc \
        = kpm.galdos(scell, erange=erange, enum=enum, estep=estep, count=count,
                     tol=tol, smooth=smooth,
                     limit=limit, kernel=kernel, rcstr_method=rcstr_method,
                     stateclass=stateclass, spr=spr,
                     omp=omp, num_threads=num_threads,
                     until=until, verbose=verbose,
                     init_adens=init_adens,  # init_aenerg=init_aenerg
                     init_acount=init_acount, init_avar=init_avar,
                     init_gdens=init_gdens,  # init_generg=init_generg
                     init_gcount=init_gcount, init_gvar=init_gvar)

    if aenerg is None and ainit:
        aenerg = ainit.x
    if generg is None and ginit:
        generg = ginit.x
    aldos = cofunc.coFunc(aenerg, adens)
    aldos.attrs.update(erange=erange, enum=enum, estep=estep,
                       tol=tol, smooth=smooth,
                       limit=limit, kernel=kernel, rcstr_method=rcstr_method,
                       stateclass=stateclass, spr=spr,
                       omp=omp, num_threads=num_threads,
                       until=until, verbose=verbose, var=avar, count=acount,
                       acc=aacc)
    gldos = cofunc.coFunc(generg, gdens)
    gldos.attrs.update(erange=erange, enum=enum, estep=estep,
                       tol=tol, smooth=smooth,
                       limit=limit, kernel=kernel, rcstr_method=rcstr_method,
                       stateclass=stateclass, spr=spr,
                       omp=omp, num_threads=num_threads,
                       until=until, verbose=verbose, var=gvar, count=gcount,
                       acc=gacc)
    return aldos, gldos


@Comliner(inmap=dict(mat='$0/scell'), outmap={0: '$0/dos'},
      preproc=dict(mat=lambda scell: scell.tbmat(format='csr')),
      shortopts=kpm.shortopts, longopts=kpm.longopts,
      opttypes=kpm.opttypes, optdoc=kpm.optdoc)
def dos(mat, rcount=None, rtol=None, rsmooth=1, limit=100, erange=10,
        enum=None, estep=None, kernel='jackson', rcstr_method='std', omp=False,
        num_threads=None, rescaled=False, until=None, verbose=False):
    """Wrapper for :func:`kpm.dos` that uses :class:`cofunc.coFunc` objects for
    input and output.
    """
    # var, count, acc?
    energ, dens \
        = kpm.dos(mat, rcount=rcount, rtol=rtol, rsmooth=rsmooth, limit=limit,
                  erange=erange, enum=enum, estep=estep,
                  kernel=kernel, rcstr_method=rcstr_method,
                  omp=omp, num_threads=num_threads, rescaled=rescaled,
                  until=until, verbose=verbose)

    dos = cofunc.coFunc(energ, dens)
    dos.attrs.update(erange=erange, enum=enum, estep=estep,
                     rcount=rcount, rtol=rtol, rsmooth=rsmooth,
                     limit=limit, kernel=kernel, rcstr_method=rcstr_method,
                     omp=omp, num_threads=num_threads,
                     until=until, verbose=verbose)
                     # var=var, acc=acc
    return dos


@Comliner(inmap=dict(scell='$0/scell', init='$0/ados'), outmap={0: '$0/ados'},
      shortopts=kpm.shortopts, longopts=kpm.longopts,
      opttypes=kpm.opttypes, optdoc=kpm.optdoc, overwrite=True)
def ados(scell, erange=10, enum=None, estep=None, count=None, tol=None,
         smooth=10, rcount=None, rtol=None, rsmooth=2, limit=100,
         kernel='jackson', rcstr_method='std', omp=False, num_threads=0,
         until=None, verbose=False, init=None):
    """Wrapper for :func:`kpm.ados` that uses :class:`cofunc.coFunc` objects
    for input and output.
    """
    #init_energ = init.x if init else None
    init_dens = init.y if init else None
    init_var = init.attrs.var if init else None
    init_count = init.attrs.count if init else None

    energ, dens, var, count, acc \
        = kpm.ados(scell, erange=erange, enum=enum, estep=estep,
                   count=count, tol=tol, smooth=smooth,
                   rcount=rcount, rtol=rtol, rsmooth=rsmooth,
                   limit=limit, kernel=kernel, rcstr_method=rcstr_method,
                   omp=omp, num_threads=num_threads,
                   until=until, verbose=verbose,
                   init_dens=init_dens,  # init_energ=init_energ
                   init_count=init_count, init_var=init_var)

    if energ is None and init:
        energ = init.x
    ados = cofunc.coFunc(energ, dens)
    ados.attrs.update(erange=erange, enum=enum, estep=estep,
                      count=count, tol=tol, smooth=smooth,
                      rcount=rcount, rtol=rtol, rsmooth=rsmooth,
                      limit=limit, kernel=kernel, rcstr_method=rcstr_method,
                      omp=omp, num_threads=num_threads,
                      until=until, verbose=verbose, var=var,
                      acc=acc)
    return ados
