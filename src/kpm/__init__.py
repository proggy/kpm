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
"""Implement the kernel polynomial method (KPM) [1].

Right now, KPM algorithms for the following target quantities are provided:

    - Density of states (DOS)
    - Local density of states (LDOS)

By averaging (arithmetic or geometric mean), also the following quantities
can be calculated:

    - Arithmetic mean of the LDOS (ALDOS)
    - Geometric mean of the LDOS (GLDOS)

The algorithms expect either tight binding matrices, or supercell definitions
as defined in the module :mod:`sc` which provide rules to create a matrix "on
the fly".

Certain submodules are written in Cython [2] to obtain better performance and
allow for OpenMP parallelization.

References:

    - [1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
    - [2] http://cython.org/
"""
__version__ = 'v0.1.0'

import itertools
import numpy
import random
import scipy
import oalg
import progmon

# import submodules
from . import disc
from . import kern
from . import misc
from . import mom
from . import rcstr
from . import resc
from . import svect
from . import dummy

try:
    from comliner import Comliner
except ImportError:
    Comliner = dummy.Decorator

# common comliner configuration
shortopts = dict(erange='e', kernel='k', rcstr='r', limit='l', omp='m',
                 num_threads='n', until='u', count='s', spr='p', smooth='o',
                 stateclass='c', enum='i', estep='t', tol='a')
longopts = dict(rcstr_method='rcstr', num_threads='num-threads')
opttypes = dict(enum=int, estep=float, num_threads=int, count=int, limit=int,
                spr=int, smooth=int, tol=float, erange=str, rtol=float,
                rcount=int, rsmooth=int)
optdoc = dict(erange='energy range', kernel='select kernel',
              rcstr_method='select reconstruction method (std, dct, ...)',
              limit='truncation limit of Chebychev expansion',
              omp='use OpenMP parallelized algorithms (if available)',
              num_threads='number of OpenMP threads to use. ' +
                          'If smaller than 1, choose automatically ' +
                          'according to the number of CPU cores',
              until='confine execution time',
              enum='number of energy discretization steps',
              estep='energy stepwidth',
              count='abort when given LDOS count has been reached ' +
                    '(sample size)',
              tol='abort when result is converged within given tolerance',
              spr='number of states per disorder realization taken into ' +
                  'account',
              smooth='set smoothness level for convergence criterion ' +
                     '(--tol)',
              stateclass='restrict selection of states to a certain state ' +
                         'class. By default, select all',
              init_dens='continue a previous calculation',
              init_var='continue a previous calculation',
              init_count='continue a previous calculation',
              verbose='with --count, show progress bar, ' +
                      'with --tol, monitor count and accuracy',
              rescaled='do not rescale, given matrix is already rescaled ' +
                       '(its spectrum fits into the interval [-1, 1])')


@Comliner(inmap=dict(mat='$0/scell'),
      preproc=dict(mat=lambda scell: scell.tbmat(format='csr')),
      outmap={0: '$0/lenerg', 1: '$0/ldens'},
      opttypes=opttypes, longopts=longopts, shortopts=shortopts,
      optdoc=optdoc)
def ldos(mat, state=0, limit=100, erange=10, enum=None, estep=None,
         kernel='jackson', rcstr_method='std', omp=False, num_threads=None,
         rescaled=False, stateclass=''):
    """Calculate local density of states (LDOS) from the given tight binding
    matrix. Return energy and density array.
    """
    if (enum and estep) or (not enum and not estep):
        raise ValueError('exactly one of "enum" and "estep" must be specified')
    mat = mat.tocsr() if isinstance(mat, scipy.sparse.base.spmatrix) \
        else scipy.sparse.csr_matrix(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError('expecting square matrix')
    rank = mat.shape[0]
    erange = (-1, 1) if rescaled else misc.symmetric_interval(erange)
    mat, params = resc.rescale(mat, erange=erange, copy=True,
                               omp=omp, num_threads=num_threads) \
        if not rescaled else (mat, (1., 0.))
    rcstr_func = rcstr.select(rcstr_method)
    kernel_func = kern.select(kernel)
    energ = _get_energ(rcstr_func, enum, estep, params)
    rcstr_args, rcstr_kwargs = _get_rcstr_args(rcstr_func, energ, enum,
                                               omp, num_threads)
    start = svect.ind(rank, state, dtype=mat.dtype)
    moments = mom.expec(mat, start, limit, omp=omp, num_threads=num_threads)
    kernel_func(moments, out=moments, omp=omp, num_threads=num_threads)
    dens = rcstr_func(moments, *rcstr_args, **rcstr_kwargs) / params[0]
    energ = resc.inverse_rescale(energ, params)
    return energ, dens


@Comliner(inmap=dict(mat='$0/scell'),
      preproc=dict(mat=lambda scell: scell.tbmat(format='csr')),
      outmap={0: '$0/lenerg', 1: '$0/ldens'},
      opttypes=opttypes, longopts=longopts, shortopts=shortopts,
      optdoc=optdoc)
def ldos_vm(mat, varmom, state=0, erange=10, enum=None, estep=None,
         kernel='jackson', rcstr_method='std', omp=False, num_threads=None,
         rescaled=False, stateclass=''):
    """Calculate local density of states (LDOS) using the given tight binding
    matrix *mat* and the energy-dependent number of Chebychev moments *varmom*.
    Return energy and density array.
    """
    if (enum and estep) or (not enum and not estep):
        raise ValueError('exactly one of "enum" or "estep" must be given')
    mat = mat.tocsr() if isinstance(mat, scipy.sparse.base.spmatrix) \
        else scipy.sparse.csr_matrix(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError('expecting square matrix')
    rank = mat.shape[0]
    erange = (-1, 1) if rescaled else misc.symmetric_interval(erange)
    mat, params = resc.rescale(mat, erange=erange, copy=True,
                               omp=omp, num_threads=num_threads) \
        if not rescaled else (mat, (1., 0.))
    rcstr_func = rcstr.select(rcstr_method)
    kernel_func = kern.select(kernel)
    energ = _get_energ(rcstr_func, enum, estep, params)
    if len(energ) != len(varmom):
        raise ValueError('length of "varmom" inconsistent with number of ' +
                         'energies')
    rcstr_args, rcstr_kwargs = _get_rcstr_args(rcstr_func, energ, enum,
                                               omp, num_threads)
    start = svect.ind(rank, state, dtype=mat.dtype)
    moments = mom.expec(mat, start, max(varmom), omp=omp,
                        num_threads=num_threads)
    ### kernel is also depending on "varmom"!
    raise NotImplementedError
    kernel_func(moments, out=moments, omp=omp, num_threads=num_threads)
    dens = rcstr_func(moments, *rcstr_args, **rcstr_kwargs) / params[0]
    energ = resc.inverse_rescale(energ, params)
    return energ, dens


@Comliner(inmap=dict(scell='$0/scell', init_dens='$0/adens', init_var='$0/avar',
                 init_count='$0/acount'),
      outmap={0: '$0/alenerg', 1: '$0/aldens', 2: '$0/alvar', 3: '$0/alcount',
              4: '$0/alacc'},
      opttypes=opttypes, longopts=longopts, shortopts=shortopts,
      optdoc=optdoc, overwrite=True)
def aldos(scell, erange=10, enum=None, estep=None, count=None, tol=None,
          smooth=10, limit=100, kernel='jackson', rcstr_method='std',
          stateclass=None, spr=1, omp=False, num_threads=0,
          until=None, verbose=False,
          init_dens=None, init_var=None, init_count=None):
    """Calculate arithmetic mean of local density of states (ALDOS).
    """

    ### compare complete parameter sets? warn when continuing calculation?
    if (count and tol) or (not count and not tol):
        raise ValueError('exactly one of "count" or "tol" must be given')
    if (enum and estep) or (not enum and not estep):
        raise ValueError('exactly one of "enum" or "estep" must be given')
    if (init_dens is not None) + (init_var is not None) \
            + (init_count is not None) not in (0, 3):
        raise ValueError('either all 3 init values have to be given or ' +
                         'None at all')

    with progmon.Abort() as abort, \
        progmon.Converge(tol=tol, smooth=smooth) as converge, \
        progmon.Until(until) as until, \
        progmon.Terminate() as term, \
        progmon.Monitor(formats=dict(acc='%g'),
                         order=['smooth', 'count', 'acc'],
                         verbose=(verbose and tol)) as mon, \
        progmon.Bar(count, text='calculate ALDOS',
                     verbose=(verbose and count)) as bar:

        # as soon as the matrix has been rescaled for the first time and a
        # rescaled energy array is retrieved from the core module, set this
        # to True
        # so this is True when at least one for-loop iteration has been done
        scaleback = False
        mean = oalg.Mean(init_value=init_dens, init_count=init_count,
                         init_var=init_var)

        # the progress bar may jump ahead if a fixed sample size is given
        if count and not abort.check() and not term.check() \
                and not until.check():
            bar.jump(mean.count)

        #size = scell.size()
        params = (1., 0.)
        energ = None

        # cycle realizations
        while not until.check() and not abort.check() \
                and (not tol or not converge.check(mean.mean())) \
                and (not count or mean.count < count) \
                and not term.check():
            mat, stateclasses = scell.tbmat(distinguish=True)
            inds = _get_stateclass(stateclass, stateclasses)
            random.shuffle(inds)
            #erange = get_erange(erange, mat)
            mat, params = resc.rescale(mat, erange=erange,
                                       omp=omp, num_threads=num_threads)
                                       ### copy=False not working???
            estep2 = estep / params[0] if estep else None

            # cycle states
            for sind in xrange(max(min(spr, len(inds)), 1)):
                if until.check() or abort.check() or term.check() \
                        or (tol and converge.check(mean.mean())) \
                        or (count and mean.count >= count):
                    break

                # calculate LDOS
                # erange isn't needed if rescaled is True
                energ, dens = ldos(mat, limit=limit, rescaled=True,
                                   rcstr_method=rcstr_method, enum=enum,
                                   estep=estep2, kernel=kernel,
                                   state=inds[sind], num_threads=num_threads)
                scaleback = True
                mean.add(dens.real/params[0])

                mon.update(smooth=smooth, count=mean.count,
                           acc=converge.delta())
                bar.step()

        if scaleback:
            energ = resc.inverse_rescale(energ, params)
        mon.set_delay(0)
        mon.update(smooth=smooth, count=mean.count, acc=converge.delta())

    for handler in (until, term, abort):
        handler.report()
    return energ, mean.mean(), mean.var(), mean.count, converge.delta()


@Comliner(inmap=dict(scell='$0/scell', init_dens='$0/gdens', init_var='$0/gvar',
                 init_count='$0/gcount'),
      outmap={0: '$0/glenerg', 1: '$0/gldens', 2: '$0/glvar', 3: '$0/glcount',
              4: '$0/glacc'},
      opttypes=opttypes, longopts=longopts, shortopts=shortopts,
      optdoc=optdoc, overwrite=True)
def gldos(scell, erange=10, enum=None, estep=None, count=None, tol=None,
          smooth=10, limit=100, kernel='jackson', rcstr_method='std',
          stateclass=None, spr=1, omp=False, num_threads=0,
          until=None, verbose=False,
          init_dens=None, init_var=None, init_count=None):
    """Calculate geometric mean of local density of states (GLDOS) (also known
    as the typical density of states).
    """

    ### compare complete parameter sets? warn when continuing calculation?
    if (count and tol) or (not count and not tol):
        raise ValueError('exactly one of "count" or "tol" must be given')
    if (enum and estep) or (not enum and not estep):
        raise ValueError('exactly one of "enum" or "estep" must be given')
    if (init_dens is not None) + (init_var is not None) \
            + (init_count is not None) not in (0, 3):
        raise ValueError('either all 3 init values have to be given or none ' +
                         'at all')

    with progmon.Abort() as abort, \
        progmon.Converge(tol=tol, smooth=smooth) as converge, \
        progmon.Until(until) as until, \
        progmon.Terminate() as term, \
        progmon.Monitor(formats=dict(acc='%g'),
                         order=['smooth', 'count', 'acc'],
                         verbose=(verbose and tol)) as mon, \
        progmon.Bar(count, text='calculate GLDOS',
                     verbose=(verbose and count)) as bar:

        # as soon as the matrix has been rescaled for the first time and a
        # rescaled energy array is retrieved from the core module, set this
        # to True
        # so this is True when at least one for-loop iteration has been done
        scaleback = False
        mean = oalg.gMean(init_value=init_dens, init_count=init_count,
                          init_var=init_var)

        # the progress bar may jump ahead if a fixed sample size is given
        if count and not abort.check() and not term.check() \
                and not until.check():
            bar.jump(mean.count)

        #size = scell.size()
        params = (1., 0.)
        energ = None

        # cycle realizations
        while not until.check() and not abort.check() \
                and (not tol or not converge.check(mean.mean())) \
                and (not count or mean.count < count) \
                and not term.check():
            mat, stateclasses = scell.tbmat(distinguish=True)
            inds = _get_stateclass(stateclass, stateclasses)
            random.shuffle(inds)
            #erange = get_erange(erange, mat)
            mat, params = resc.rescale(mat, erange=erange,
                                       omp=omp, num_threads=num_threads)
                                       ### copy=False not working???
            estep2 = estep / params[0] if estep else None

            # cycle states
            for sind in xrange(max(min(spr, len(inds)), 1)):
                if until.check() or abort.check() or term.check() \
                        or (tol and converge.check(mean.mean())) \
                        or (count and mean.count >= count):
                    break

                # calculate LDOS
                # erange isn't needed if rescaled is True
                energ, dens = ldos(mat, limit=limit, rescaled=True,
                                   rcstr_method=rcstr_method, enum=enum,
                                   estep=estep2, kernel=kernel,
                                   state=inds[sind],
                                   num_threads=num_threads)
                scaleback = True
                rdens = dens.real / params[0]
                eps = numpy.finfo(float).eps
                mean.add(rdens.clip(eps))  # because sometimes small
                                            # negative values occur

                mon.update(smooth=smooth, count=mean.count,
                           acc=converge.delta())
                bar.step()

        if scaleback:
            energ = resc.inverse_rescale(energ, params)
        mon.set_delay(0)
        mon.update(smooth=smooth, count=mean.count, acc=converge.delta())

    for handler in (until, term, abort):
        handler.report()
    return energ, mean.mean(), mean.var(), mean.count, converge.delta()


@Comliner(inmap=dict(scell='$0/scell', init_adens='$0/adens',
                 init_avar='$0/avar', init_acount='$0/acount',
                 init_gdens='$0/gdens',
                 init_gvar='$0/gvar', init_gcount='$0/gcount'),
      outmap={0: '$0/alenerg', 1: '$0/aldens', 2: '$0/alvar', 3: '$0/alcount',
              4: '$0/alacc',
              5: '$0/glenerg', 6: '$0/gldens', 7: '$0/glvar', 8: '$0/glcount',
              9: '$0/glacc'},
      opttypes=opttypes, longopts=longopts, shortopts=shortopts,
      optdoc=optdoc, overwrite=True)
def galdos(scell, erange=10, enum=None, estep=None, count=None, tol=None,
           smooth=10, limit=100, kernel='jackson', rcstr_method='std',
           stateclass=None, spr=1, omp=False, num_threads=0,
           until=None, verbose=False,
           init_adens=None, init_avar=None, init_acount=None,
           init_gdens=None, init_gvar=None, init_gcount=None):
    """Calculate both the geometric (typical average) and the arithmetic mean
    of the local density of states (GLDOS and ALDOS) at the same time, using
    each local density twice.  In this way, the numerical effort is easily
    reduced by a factor of 2 if both types of averages are needed.
    """

    ### compare complete parameter sets? warn when continuing calculation?
    if (count and tol) or (not count and not tol):
        raise ValueError('exactly one of "count" or "tol" must be given')
    if (enum and estep) or (not enum and not estep):
        raise ValueError('exactly one of "enum" or "estep" must be given')
    if (init_adens is not None) + (init_avar is not None) \
            + (init_acount is not None) not in (0, 3):
        raise ValueError('either all 3 init values have to be given or none ' +
                         'at all')
    if (init_gdens is not None) + (init_gvar is not None) \
            + (init_gcount is not None) not in (0, 3):
        raise ValueError('either all 3 init values have to be given or none ' +
                         'at all')

    with progmon.Abort() as abort, \
        progmon.Converge(tol=tol, smooth=smooth) as aconverge, \
        progmon.Converge(tol=tol, smooth=smooth) as gconverge, \
        progmon.Until(until) as until, \
        progmon.Terminate() as term, \
        progmon.Monitor(formats=dict(aacc='%g', gacc='%g'),
                         order=['smooth', 'acount', 'aacc', 'gcount', 'gacc'],
                         verbose=(verbose and tol)) as mon, \
        progmon.Bar(count, text='calculate ALDOS and GLDOS',
                     verbose=(verbose and count)) as bar:

        # as soon as the matrix has been rescaled for the first time and a
        # rescaled energy array is retrieved from the core module, set this
        # to True
        # so this is True when at least one for-loop iteration has been done
        scaleback = False

        amean = oalg.Mean(init_value=init_adens, init_count=init_acount,
                          init_var=init_avar)
        gmean = oalg.gMean(init_value=init_gdens, init_count=init_gcount,
                           init_var=init_gvar)

        # the progress bar may jump ahead if a fixed sample size is given
        if count and not abort.check() and not term.check() \
                and not until.check():
            bar.jump(min(amean.count, gmean.count))

        #size = scell.size()
        params = (1., 0.)
        energ = None

        # cycle realizations
        while not until.check() and not abort.check() \
                and (not tol or not aconverge.check(amean.mean())
                     or not gconverge.check(gmean.mean())) \
                and (not count or amean.count < count or gmean.count < count) \
                and not term.check():
            mat, stateclasses = scell.tbmat(distinguish=True)
            inds = _get_stateclass(stateclass, stateclasses)
            random.shuffle(inds)
            #erange = get_erange(erange, mat)
            mat, params = resc.rescale(mat, erange=erange,
                                       omp=omp, num_threads=num_threads)
                                       ### copy=False not working???
            estep2 = estep / params[0] if estep else None

            # cycle states
            for sind in xrange(max(min(spr, len(inds)), 1)):
                if until.check() or abort.check() or term.check() \
                    or (tol and aconverge.check(amean.mean())
                        and gconverge.check(gmean.mean())) \
                    or (count and amean.count >= count
                        and gmean.count >= count):
                    break

                # calculate LDOS
                # erange isn't needed if rescaled is True
                energ, dens = ldos(mat, limit=limit, rescaled=True,
                                   rcstr_method=rcstr_method, enum=enum,
                                   estep=estep2, kernel=kernel,
                                   state=inds[sind], num_threads=num_threads)
                scaleback = True
                rdens = dens.real / params[0]
                amean.add(rdens)
                eps = numpy.finfo(float).eps
                gmean.add(rdens.clip(eps))  # because sometimes small negative
                                            # values occur

                mon.update(smooth=smooth, acount=amean.count,
                           aacc=aconverge.delta(),
                           gcount=gmean.count, gacc=gconverge.delta())
                bar.step()

        if scaleback:
            energ = resc.inverse_rescale(energ, params)
        mon.set_delay(0)
        mon.update(smooth=smooth, acount=amean.count, aacc=aconverge.delta(),
                   gcount=gmean.count, gacc=gconverge.delta())

    for handler in (until, term, abort):
        handler.report()
    return energ, amean.mean(), amean.var(), amean.count, aconverge.delta(), \
        energ, gmean.mean(), gmean.var(), gmean.count, gconverge.delta()


@Comliner(inmap=dict(mat='$0/scell'),
      preproc=dict(mat=lambda scell: scell.tbmat(format='csr')),
      outmap={0: '$0/energ', 1: '$0/dens'},
      opttypes=opttypes, longopts=longopts, shortopts=shortopts,
      optdoc=optdoc)
def dos(mat, rcount=None, rtol=None, rsmooth=2, limit=100, erange=10,
        enum=None, estep=None, kernel='jackson', rcstr_method='std', omp=False,
        num_threads=None, rescaled=False, until=None, verbose=False):
    """Calculate density of states (DOS) using "stochastic evaluation of
    traces".

    Note that there is no ensemble averaging. Use the function ados to include
    an average over different disorder configurations if a random system is
    studied.
    """
    if (rcount and rtol) or (not rcount and not rtol):
        raise ValueError('exactly one of "rcount" or "rtol" must be given')
    if (enum and estep) or (not enum and not estep):
        raise ValueError('exactly one of "enum" or "estep" must be given')
    mat = mat.tocsr() if isinstance(mat, scipy.sparse.base.spmatrix) \
        else scipy.sparse.csr_matrix(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError('expecting square matrix')
    rank = mat.shape[0]
    erange = (-1, 1) if rescaled else misc.symmetric_interval(erange)
    mat, params = resc.rescale(mat, erange=erange, copy=True,
                               omp=omp, num_threads=num_threads) \
        if not rescaled else (mat, (1., 0.))
    rcstr_func = rcstr.select(rcstr_method)
    kernel_func = kern.select(kernel)
    energ = _get_energ(rcstr_func, enum, estep, params)
    rcstr_args, rcstr_kwargs = _get_rcstr_args(rcstr_func, energ, enum,
                                               omp, num_threads)

    with progmon.Abort() as abort, \
        progmon.Converge(tol=rtol, smooth=rsmooth) as converge, \
        progmon.Until(until) as until, \
        progmon.Terminate() as term, \
        progmon.Monitor(formats=dict(acc='%g'),
                         order=['smooth', 'count', 'acc'],
                         verbose=(verbose and rtol)) as mon, \
        progmon.Bar(rcount, text='calculate DOS',
                     verbose=(verbose and rcount)) as bar:

        mean = oalg.Mean()

        # cycle random-phase start vectors
        while not until.check() and not abort.check() \
                and (not rtol or not converge.check(mean.mean())) \
                and (not rcount or mean.count < rcount) \
                and not term.check():
            start = svect.randphase(rank, dtype=complex)  # dtype=complex
            m = mom.expec(mat, start, limit, omp=omp, num_threads=num_threads)
            mean.add(m)

            mon.update(smooth=rsmooth, count=mean.count, acc=converge.delta())
            bar.step()

        moments = mean.mean()  # / rank  # or not?
        kernel_func(moments, out=moments, omp=omp, num_threads=num_threads)
        dens = rcstr_func(moments, *rcstr_args, **rcstr_kwargs)
        energ = resc.inverse_rescale(energ, params)

        # WTF, let's normalize the DOS by hand...
        area = numpy.trapz(dens, energ)
        #print(area)
        dens /= area
        #print(numpy.trapz(dens, energ))

        mon.set_delay(0)
        mon.update(smooth=rsmooth, count=mean.count, acc=converge.delta())

    for handler in (until, term, abort):
        handler.report()
    return energ, dens  # how to propagate error?


@Comliner(inmap=dict(scell='$0/scell', init_dens='$0/adens', init_var='$0/avar',
                 init_count='$0/acount'),
      outmap={0: '$0/aenerg', 1: '$0/adens', 2: '$0/avar', 3: '$0/acount',
              4: '$0/aacc'},
      opttypes=opttypes, longopts=longopts, shortopts=shortopts,
      optdoc=optdoc, overwrite=True)
def ados(scell, erange=10., enum=None, estep=None, count=None, tol=None,
         smooth=10, rcount=None, rtol=None, rsmooth=2, limit=100,
         kernel='jackson', rcstr_method='std', omp=False, num_threads=0,
         until=None, verbose=False,
         init_dens=None, init_var=None, init_count=None):
    """Calculate arithmetic mean of density of states (ADOS) (ensemble
    average).
    """

    ### compare complete parameter sets? warn when continuing calculation?
    if (count and tol) or (not count and not tol):
        raise ValueError('exactly one of "count" or "tol" must be given')
    if (enum and estep) or (not enum and not estep):
        raise ValueError('exactly one of "enum" or "estep" must be given')
    if (init_dens is not None) + (init_var is not None) \
            + (init_count is not None) not in (0, 3):
        raise ValueError('either all 3 init values have to be given or ' +
                         'none at all')

    with progmon.Abort() as abort, \
        progmon.Converge(tol=tol, smooth=smooth) as converge, \
        progmon.Until(until) as until, \
        progmon.Terminate() as term, \
        progmon.Monitor(formats=dict(acc='%g'),
                         order=['smooth', 'count', 'acc'],
                         verbose=(verbose and tol)) as mon, \
        progmon.Bar(count, text='calculate ADOS',
                     verbose=(verbose and count)) as bar:

        # as soon as the matrix has been rescaled for the first time and a
        # rescaled energy array is retrieved from the core module, set this
        # to True
        # so this is True when at least one for-loop iteration has been done
        scaleback = False
        mean = oalg.Mean(init_value=init_dens, init_count=init_count,
                         init_var=init_var)

        # the progress bar may jump ahead if a fixed sample size is given
        if count and not abort.check() and not term.check() \
                and not until.check():
            bar.jump(mean.count)

        #size = scell.size()
        params = (1., 0.)
        energ = None

        # cycle realizations
        while not until.check() and not abort.check() \
                and (not tol or not converge.check(mean.mean())) \
                and (not count or mean.count < count) \
                and not term.check():
            mat = scell.tbmat()
            #erange = get_erange(erange, mat)
            mat, params = resc.rescale(mat, erange=erange,
                                       omp=omp, num_threads=num_threads)
                                       ### copy=False not working???
            estep2 = estep / params[0] if estep else None

            # calculate LDOS
            # erange isn't needed if rescaled is True
            energ, dens = dos(mat, limit=limit, rescaled=True,
                              rcstr_method=rcstr_method, enum=enum,
                              estep=estep2, kernel=kernel,
                              rcount=rcount, rtol=rtol, rsmooth=rsmooth,
                              omp=omp, num_threads=num_threads)
            scaleback = True
            mean.add(dens.real / params[0])

            mon.update(smooth=smooth, count=mean.count, acc=converge.delta())
            bar.step()

        if scaleback:
            energ = resc.inverse_rescale(energ, params)
        mon.set_delay(0)
        mon.update(smooth=smooth, count=mean.count, acc=converge.delta())

    for handler in (until, term, abort):
        handler.report()
    return energ, mean.mean(), mean.var(), mean.count, converge.delta()


def _get_stateclass(stateclass_option, stateclasses):
    """Return state indices according to the selected stateclass(es).
    """
    if stateclass_option is None or not stateclass_option:
        # by default, return all available indices
        return list(set(itertools.chain(*stateclasses)))
    else:
        classinds = stateclass_option.split(',')
        indices = []
        for classind in classinds:
            indices.append(int(classind))
        return list(set(indices))


def _get_energ(rcstr_func, enum, estep, params):
    if rcstr_func is rcstr.dct:
        if estep:
            raise ValueError('setting energy stepwidth not possible with DCT')
        energ = disc.cosine(enum)
    elif rcstr_func is rcstr.std:
        energ = disc.uniform(enum or estep / params[0])
    else:
        raise ValueError('unknown reconstruction method: %s'
                         % rcstr_func.__name__)
    return energ


def _get_rcstr_args(rcstr_func, energ, enum, omp, num_threads):
    rcstr_args = []
    rcstr_kwargs = dict(omp=omp, num_threads=num_threads)
    if rcstr_func is rcstr.std:
        rcstr_args.append(energ)  # use uniform discretization in STD mode
    elif rcstr_func is rcstr.dct:
        rcstr_kwargs['ndisc'] = enum  # pass number of intervals
    return rcstr_args, rcstr_kwargs
