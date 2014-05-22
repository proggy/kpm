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
"""Implement the kernel polynomial method (KPM) [1].

Right now, KPM algorithms for the following target quantities are provided:
- density of states (DOS)
- local density of states (LDOS)

By averaging (arithmetic or geometric mean), also the following quantities
can be calculated:
- total local density of states (arithmetic mean of LDOS)
- typical local density of states (geometric mean of LDOS)

The algorithms expect either tight binding matrices, or supercell definitions
as defined in the module "sc" which define the rules to create a matrix "on the
fly".

Certain submodules are written in Cython [2] to obtain better performance and
allow for OpenMP parallelization.

[1] Weiße et al., Rev. Mod. Phys. 78, 275 (2006)
[2] http://cython.org/

To do:
--> implement other reconstruction methods (fct, fft, dft)
--> by choice, return cofunc objects
--> implement the "stochastical method" to compute ADOS
--> keep rescaling factors

"""
__created__ = '2013-07-06'
__modified__ = '2014-02-28'
# former tb.kpm2 (developed 2012-08-06 until 2013-06-25)
# based on tb.kpm, developed from 2011-10-11 until 2012-06-27,
# which itself is the former tbc.pyx from 2011-04-14 until 2011-05-17
import itertools
import numpy
import random
import scipy
import h5obj.tools
import oalg
import progress
import dummy

# import submodules
import disc
import kern
import misc
import mom
import rcstr
import resc
import svect

try:
    from frog import Frog
except ImportError:
    Frog = dummy.Decorator


# algorithms directly interfacing fast Cython implementations
# - ldos
# - dos

# algorithms that use LDOS as an input
# - aldos
# - gldos
# - galdos

# algorithms that use DOS as an input
# - ados


# common frog configuration
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


@Frog(inmap=dict(mat='$0/scell'),
      preproc=dict(mat=lambda scell: scell.tbmat(format='csr')),
      outmap={0: '$0/lenerg', 1: '$0/ldens'},
      opttypes=opttypes, longopts=longopts, shortopts=shortopts,
      optdoc=optdoc)
def ldos(mat, state=0, limit=100, erange=10, enum=None, estep=None,
         kernel='jackson', rcstr_method='std', omp=False, num_threads=None,
         rescaled=False, stateclass=''):
    """Calculate local density of states (LDOS) from the given tight binding
    matrix. Return energy and density array."""
    # 2013-07-06 until 2014-01-14
    # former tb.kpm2._ldos (developed 2012-08-25 until 2013-06-25)
    # former tb.kpm._Ldos (developed 2011-11-23 until 2012-03-14)
    # former tb._Ldos from 2011-02-20 until 2011-05-10
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


@Frog(inmap=dict(mat='$0/scell'),
      preproc=dict(mat=lambda scell: scell.tbmat(format='csr')),
      outmap={0: '$0/lenerg', 1: '$0/ldens'},
      opttypes=opttypes, longopts=longopts, shortopts=shortopts,
      optdoc=optdoc)
def ldos_vm(mat, varmom, state=0, erange=10, enum=None, estep=None,
         kernel='jackson', rcstr_method='std', omp=False, num_threads=None,
         rescaled=False, stateclass=''):
    """Calculate local density of states (LDOS) using the given tight binding
    matrix "mat" and the energy-dependent number of Chebychev moments "varmom".
    Return energy and density array."""
    # 2013-07-06 until 2014-01-14
    # former tb.kpm2._ldos (developed 2012-08-25 until 2013-06-25)
    # former tb.kpm._Ldos (developed 2011-11-23 until 2012-03-14)
    # former tb._Ldos from 2011-02-20 until 2011-05-10
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


@Frog(inmap=dict(scell='$0/scell', init_dens='$0/adens', init_var='$0/avar',
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
    """Calculate arithmetic mean of local density of states."""
    # 2013-07-22 until 2013-07-24
    # based on tb.kpm2._aldos (developed 2012-09-03 until 2013-06-20)
    # based on tb.kpm._Ados (developed from 2012-04-26 until 2012-07-16) and
    #          tb.kpm._Gdos (developed from 2011-12-20 until 2012-07-17)
    # based on tb.kpm._Gdos from 2011-12-20 until 2012-04-18

    ### compare complete parameter sets? warn when continuing calculation?
    if (count and tol) or (not count and not tol):
        raise ValueError('exactly one of "count" or "tol" must be given')
    if (enum and estep) or (not enum and not estep):
        raise ValueError('exactly one of "enum" or "estep" must be given')
    if (init_dens is not None) + (init_var is not None) \
            + (init_count is not None) not in (0, 3):
        raise ValueError('either all 3 init values have to be given or ' +
                         'None at all')

    with progress.Abort() as abort, \
        progress.Converge(tol=tol, smooth=smooth) as converge, \
        progress.Until(until) as until, \
        progress.Terminate() as term, \
        progress.Monitor(formats=dict(acc='%g'),
                         order=['smooth', 'count', 'acc'],
                         verbose=(verbose and tol)) as mon, \
        progress.Bar(count, text='calculate ALDOS',
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


@Frog(inmap=dict(scell='$0/scell', init_dens='$0/gdens', init_var='$0/gvar',
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
    """Calculate geometric mean of local density of states (known as the
    typical density of states)."""
    # 2013-07-24 - 2013-07-29
    # based on tb2.kpm.aldos (developed 2013-07-22 until 2013-07-24)
    # based on tb.kpm2._aldos (developed 2012-09-03 until 2013-06-20)
    # based on tb.kpm._Ados (developed from 2012-04-26 until 2012-07-16) and
    #          tb.kpm._Gdos (developed from 2011-12-20 until 2012-07-17)
    # based on tb.kpm._Gdos from 2011-12-20 until 2012-04-18

    ### compare complete parameter sets? warn when continuing calculation?
    if (count and tol) or (not count and not tol):
        raise ValueError('exactly one of "count" or "tol" must be given')
    if (enum and estep) or (not enum and not estep):
        raise ValueError('exactly one of "enum" or "estep" must be given')
    if (init_dens is not None) + (init_var is not None) \
            + (init_count is not None) not in (0, 3):
        raise ValueError('either all 3 init values have to be given or none ' +
                         'at all')

    with progress.Abort() as abort, \
        progress.Converge(tol=tol, smooth=smooth) as converge, \
        progress.Until(until) as until, \
        progress.Terminate() as term, \
        progress.Monitor(formats=dict(acc='%g'),
                         order=['smooth', 'count', 'acc'],
                         verbose=(verbose and tol)) as mon, \
        progress.Bar(count, text='calculate GLDOS',
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


@Frog(inmap=dict(scell='$0/scell', init_adens='$0/adens',
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
    """Calculate both geometric (typical average) and arithmetic mean of the
    local density of states at the same time, using each local density twice.
    In this way, the numerical effort is easily reduced by a factor of 2 if
    both types of averages are needed."""
    # 2013-07-24 until 2013-07-29
    # based on tb2.kpm.gldos (developed 2013-07-24 until 2013-07-24)
    # based on tb2.kpm.aldos (developed 2013-07-22 until 2013-07-24)
    # based on tb.kpm2._aldos (developed 2012-09-03 until 2013-06-20)
    # based on tb.kpm._Ados (developed from 2012-04-26 until 2012-07-16) and
    #          tb.kpm._Gdos (developed from 2011-12-20 until 2012-07-17)
    # based on tb.kpm._Gdos from 2011-12-20 until 2012-04-18

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

    with progress.Abort() as abort, \
        progress.Converge(tol=tol, smooth=smooth) as aconverge, \
        progress.Converge(tol=tol, smooth=smooth) as gconverge, \
        progress.Until(until) as until, \
        progress.Terminate() as term, \
        progress.Monitor(formats=dict(aacc='%g', gacc='%g'),
                         order=['smooth', 'acount', 'aacc', 'gcount', 'gacc'],
                         verbose=(verbose and tol)) as mon, \
        progress.Bar(count, text='calculate ALDOS and GLDOS',
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


@Frog(inmap=dict(mat='$0/scell'),
      preproc=dict(mat=lambda scell: scell.tbmat(format='csr')),
      outmap={0: '$0/energ', 1: '$0/dens'},
      opttypes=opttypes, longopts=longopts, shortopts=shortopts,
      optdoc=optdoc)
def dos(mat, rcount=None, rtol=None, rsmooth=2, limit=100, erange=10,
        enum=None, estep=None, kernel='jackson', rcstr_method='std', omp=False,
        num_threads=None, rescaled=False, until=None, verbose=False):
    """Calculate density of states using "stochastic evaluation of traces".

    Note that there is no ensemble averaging. Use the function ados to include
    an average over different disorder configurations.

    """
    # 2014-01-14
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

    with progress.Abort() as abort, \
        progress.Converge(tol=rtol, smooth=rsmooth) as converge, \
        progress.Until(until) as until, \
        progress.Terminate() as term, \
        progress.Monitor(formats=dict(acc='%g'),
                         order=['smooth', 'count', 'acc'],
                         verbose=(verbose and rtol)) as mon, \
        progress.Bar(rcount, text='calculate DOS',
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
        #print area
        dens /= area
        #print numpy.trapz(dens, energ)

        mon.set_delay(0)
        mon.update(smooth=rsmooth, count=mean.count, acc=converge.delta())

    for handler in (until, term, abort):
        handler.report()
    return energ, dens  # how to propagate error?


@Frog(inmap=dict(scell='$0/scell', init_dens='$0/adens', init_var='$0/avar',
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
    """Calculate arithmetic mean of density of states (ensemble average)."""
    # 2014-01-13

    ### compare complete parameter sets? warn when continuing calculation?
    if (count and tol) or (not count and not tol):
        raise ValueError('exactly one of "count" or "tol" must be given')
    if (enum and estep) or (not enum and not estep):
        raise ValueError('exactly one of "enum" or "estep" must be given')
    if (init_dens is not None) + (init_var is not None) \
            + (init_count is not None) not in (0, 3):
        raise ValueError('either all 3 init values have to be given or ' +
                         'none at all')

    with progress.Abort() as abort, \
        progress.Converge(tol=tol, smooth=smooth) as converge, \
        progress.Until(until) as until, \
        progress.Terminate() as term, \
        progress.Monitor(formats=dict(acc='%g'),
                         order=['smooth', 'count', 'acc'],
                         verbose=(verbose and tol)) as mon, \
        progress.Bar(count, text='calculate ADOS',
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


#===============================================================#
# Helper functions to get count or accuracy from GLDOS or ALDOS #
#===============================================================#


@Frog()
def glcount(filename):
    """Get the attribute "gldos.attrs.count" from the dataset "gldos" from the
    given HDF5 file."""
    # 2013-07-28
    try:
        return h5obj.tools.h5load(filename + '/gldos').attrs.count
    except:
        return None


@Frog()
def alcount(filename):
    """Get the attribute "aldos.attrs.count" from the dataset "aldos" from the
    given HDF5 file."""
    # 2013-07-28
    try:
        return h5obj.tools.h5load(filename + '/aldos').attrs.count
    except:
        return None


@Frog()
def glacc(filename):
    """Get the attribute "gldos.attrs.acc" from the dataset "gldos" from the
    given HDF5 file."""
    # 2013-07-28
    try:
        return h5obj.tools.h5load(filename + '/gldos').attrs.acc
    except:
        return None


@Frog()
def alacc(filename):
    """Get the attribute "aldos.attrs.acc" from the dataset "aldos" from the
    given HDF5 file."""
    # 2013-07-28
    try:
        return h5obj.tools.h5load(filename + '/aldos').attrs.acc
    except:
        return None


@Frog()
def gllimit(filename):
    """Get the attribute "gldos.attrs.limit" from the dataset "gldos" from the
    given HDF5 file."""
    # 2014-01-07
    try:
        return h5obj.tools.h5load(filename + '/gldos').attrs.limit
    except:
        return None


@Frog()
def allimit(filename):
    """Get the attribute "aldos.attrs.limit" from the dataset "aldos" from the
    given HDF5 file."""
    # 2014-01-14
    try:
        return h5obj.tools.h5load(filename + '/aldos').attrs.limit
    except:
        return None


@Frog()
def acount(filename):
    """Get the attribute "ados.attrs.count" from the dataset "ados" from the
    given HDF5 file."""
    # 2014-01-14
    try:
        return h5obj.tools.h5load(filename + '/ados').attrs.count
    except:
        return None


@Frog()
def aacc(filename):
    """Get the attribute "ados.attrs.acc" from the dataset "ados" from the
    given HDF5 file."""
    # 2014-01-14
    try:
        return h5obj.tools.h5load(filename + '/ados').attrs.acc
    except:
        return None


@Frog()
def alimit(filename):
    """Get the attribute "ados.attrs.limit" from the dataset "ados" from the
    given HDF5 file."""
    # 2014-01-14
    try:
        return h5obj.tools.h5load(filename + '/ados').attrs.limit
    except:
        return None


@Frog()
def lcount(filename):
    """Get the attribute "ldos.attrs.count" from the dataset "ldos" from the
    given HDF5 file."""
    # 2014-01-14
    try:
        return h5obj.tools.h5load(filename + '/ldos').attrs.count
    except:
        return None


@Frog()
def lacc(filename):
    """Get the attribute "ldos.attrs.acc" from the dataset "ldos" from the
    given HDF5 file."""
    # 2014-01-14
    try:
        return h5obj.tools.h5load(filename + '/ldos').attrs.acc
    except:
        return None


@Frog()
def llimit(filename):
    """Get the attribute "ldos.attrs.limit" from the dataset "ldos" from the
    given HDF5 file."""
    # 2014-01-14
    try:
        return h5obj.tools.h5load(filename + '/ldos').attrs.limit
    except:
        return None


@Frog()
def dcount(filename):
    """Get the attribute "dos.attrs.count" from the dataset "dos" from the
    given HDF5 file."""
    # 2014-01-14
    try:
        return h5obj.tools.h5load(filename + '/dos').attrs.count
    except:
        return None


@Frog()
def dacc(filename):
    """Get the attribute "dos.attrs.acc" from the dataset "dos" from the
    given HDF5 file."""
    # 2014-01-14
    try:
        return h5obj.tools.h5load(filename + '/dos').attrs.acc
    except:
        return None


@Frog()
def dlimit(filename):
    """Get the attribute "dos.attrs.limit" from the dataset "dos" from the
    given HDF5 file."""
    # 2014-01-14
    try:
        return h5obj.tools.h5load(filename + '/dos').attrs.limit
    except:
        return None


@Frog()
def glstderr0(filename):
    """Return the standard error of the GLDOS at zero energy.
    """
    # 2014-02-28
    try:
        gldos = h5obj.tools.h5load(filename+'/gldos')
        var = gldos.a2cf('var')
        var0 = var(0.)
        count = gldos.attrs.count
        stderr0 = numpy.sqrt(var0/(count-1))
        return stderr0
    except:
        return None


@Frog()
def alstderr0(filename):
    """Return the standard error of the ALDOS at zero energy.
    """
    # 2014-02-28
    try:
        aldos = h5obj.tools.h5load(filename+'/aldos')
        var = aldos.a2cf('var')
        var0 = var(0.)
        count = aldos.attrs.count
        stderr0 = numpy.sqrt(var0/(count-1))
        return stderr0
    except:
        return None


@Frog()
def astderr0(filename):
    """Return the standard error of the ADOS at zero energy.
    """
    # 2014-02-28
    try:
        ados = h5obj.tools.h5load(filename+'/ados')
        var = ados.a2cf('var')
        var0 = var(0.)
        count = ados.attrs.count
        stderr0 = numpy.sqrt(var0/(count-1))
        return stderr0
    except:
        return None


@Frog(inmap=dict(gldos_list='$@/gldos', aldos_list='$@/aldos'))
def checksigma(gldos_list, aldos_list):
    """Investigate the fluctuation of the standard error of the geometric mean
    among independent calculations."""
    # 2014-02-28
    gvals = []
    avals = []
    gstderrs = []
    astderrs = []
    gammavals = []
    gammastds = []
    print
    print 'GLDOS                ALDOS                Gamma'
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

        print '%.7f+-%.7f %.7f+-%.7f %.7f+-%.7f' \
            % (gldos0val, gldos0stderr, aldos0val, aldos0stderr,
               gamma0val, gamma0std)

    # compare with standard deviation
    gst = numpy.std(gvals)
    ast = numpy.std(avals)
    gammast = numpy.std(gammavals)
    print 'STD: %.7f            %.7f            %.7f' % (gst, ast, gammast)


@Frog()
def glval0(filename):
    """Return the GLDOS at zero energy.
    """
    # 2014-02-28
    try:
        gldos = h5obj.tools.h5load(filename+'/gldos')
        return gldos(0.)
    except:
        return None


@Frog()
def alval0(filename):
    """Return the ALDOS at zero energy.
    """
    # 2014-02-28
    try:
        aldos = h5obj.tools.h5load(filename+'/aldos')
        return aldos(0.)
    except:
        return None


@Frog()
def aval0(filename):
    """Return the ADOS at zero energy.
    """
    # 2014-02-28
    try:
        ados = h5obj.tools.h5load(filename+'/ados')
        return ados(0.)
    except:
        return None


#class _ldos_many(_ldos):
  #"""Calculate local density of states (LDOS) using the kernel polynomial
  #method (KPM) [1] for many truncation limits at once.

  #[1] Weisse et al., Rev. Mod. Phys. 78, 275 (2006)"""
  #__created__ = '2013-05-31'
  #__modified__ = '2013-06-19'
  #outnames = ['densities']

  #def main(self):
    ## get matrix
    #if isinstance(self.infile.__scell__, tb.sc.SuperCell):
      #mat, stateclasses = self.infile.__scell__.tbmat(distinguish=True)
      #mat = mat.tocsr()
      #inds = self.get_stateclass(self.opts.stateclass, stateclasses)
      #size = self.infile.__scell__.size()
    #elif isinstance(self.infile.__scell__, scipy.sparse.base.spmatrix):
      #mat = self.infile.__scell__.tocsr()
      #inds = None
      #size = None # use rank of the matrix?
    #else:
      ## maybe a dense matrix is given?
      #mat = scipy.sparse.csr_matrix(self.infile.__scell__)
      #inds = None
      #size = None # use rank of the matrix?

    ## get matrix dimensions (assume square matrix)
    #if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
      #raise ValueError, 'bad matrix: expecting square matrix, not %ix%i' \
                        #% mat.shape
    #rank = mat.shape[0]
    #if inds is None:
      #inds = range(rank)

    ## select states
    #states = self.get_states(self.opts.state, inds)

    ## get truncation limits
    #limits = self.get_limits(self.opts.limit, size)

    ## get energy range
    #erange = self.get_erange(self.opts.erange, mat)

    ## determine energy range and rescale hamiltonian matrix
    #mat, params = resc.rescale(mat, erange=erange, copy=True,
                               #omp=self.opts.omp) \
                  #if not self.opts.rescaled else (mat, (1., 0.))

    ## select reconstruction method (only DCT or STD right now)
    #selected_rcstr_method = rcstr.select(self.opts.rcstr, many=True)

    ## get number of intervals
    ##intervals = self.get_intervals(self.opts.intervals, limit, params)
    #### for each it's own intervals?

    ## discretize energy axis
    #if selected_rcstr_method is rcstr.dct_many:
      #energ = disc.cosine(self.opts.intervals)
    #elif selected_rcstr_method is rcstr.std_many:
      #energ = disc.uniform(self.opts.intervals)
    #else:
      #print >>sys.stderr, 'ldos: unknown reconstruction method: %s' \
                          #% selected_rcstr_method
      #return

    ## select keyword arguments for reconstruction algorithm
    #rcstr_args = []
    #rcstr_kwargs = dict(omp=self.opts.omp, limits=limits)
    #if selected_rcstr_method is rcstr.std_many:
      #rcstr_args.append(energ) # use uniform discretization in STD mode
    #elif selected_rcstr_method is rcstr.dct_many:
      #rcstr_kwargs['ndisc'] = self.opts.intervals # pass number of intervals

    ## select kernel
    #selected_kernel = kern.select(self.opts.kernel)

    ## cycle states, average over local densities
    #means = [oalg.Mean() for limit in limits]
    #bar = progress.Bar(len(states)*len(limits), verbose=self.opts.verbose)
    #for state in states:
      ## create starting vector
      #start = svect.ind(rank, state, dtype=mat.dtype)

      ## calculate Chebychev moments, apply kernel
      #moments = mom.expec(mat, start, limit, omp=self.opts.omp)
      #selected_kernel(moments, out=moments, omp=self.opts.omp)

      ## reconstruct target function (local density of states)
      #denses = selected_rcstr_method(moments, *rcstr_args, **rcstr_kwargs)

      ## add data to online-averaging object
      #for mean, dens in zip(means, denses):
        #mean.add(dens.real/params[0])
        #bar.step()
    #bar.end()

    ## inverse-rescale energy axis
    #energ = resc.inverse_rescale(energ, params)

    ## save result
    #densities = []
    #for mean, limit in zip(means, limits):
      #ldos = cofunc.coFunc(energ, mean.mean())
      #ldos.attrs.limit  = limit
      #ldos.attrs.states = states
      #ldos.attrs.count  = mean.count
      #ldos.attrs.var    = mean.var()
      #ldos.attrs.ci     = mean.ci()
      #ldos.attrs.kernel = selected_kernel.__name__
      #ldos.attrs.rcstr  = selected_rcstr_method.__name__
      #ldos.attrs.erange = erange
      #ldos.attrs.intervals = intervals*params[0] \
                            #if isinstance(intervals, float) else intervals
      #ldos.attrs.ndisc     = len(energ)
      #densities.append(ldos)
    #self.outfile.densities = densities
    ##self.outfile.limits    = limits


#class _aldos_many(AveragingLDOS_KPMHDP00i):
  #"""Calculate arithmetic mean of local density of states for many different
  #truncation numbers at once.

  #All results will be saved into the same file, named "aldos100", "aldos200",
  #etc. according to the chosen truncation numbers."""
  #__created__ = '2013-06-01'
  #__modified__ = '2013-06-19'
  ## based on tb.kpm2._aldos (developed 2012-09-03 until 2012-10-09)
  ## based on tb.kpm._Ados (developed from 2012-04-26 until 2012-07-16) and on
  ## tb.kpm._Gdos (developed from 2011-12-20 until 2012-07-17)
  ## based on tb.kpm._Gdos from 2011-12-20 until 2012-04-18
  #innames = ['__scell__']
  #outnames = [] # names will be determined later (aldos100, aldos500, ...)
  #__autoinst__ = True

  #def prepare(self):
    ## get truncation limit
    ##size = self.infile.__scell__.size() # how to enable this?
    #self.limits_from_option = self.get_limits(self.opts.limit) # size

    ## determine outnames
    #self.outnames = ['aldos%i' % limit for limit in self.limits_from_option]

  #def main(self):
    ## make sure a supercell definition is given
    #if not isinstance(self.infile.__scell__, tb.sc.SuperCell):
      #raise ValueError, 'expecting supercell definition'

    ## instantiate LDOS HDP
    #ldosfunc = _ldos_many()

    ## understand sample size option
    #ssize, tol, digits = tb.misc.get_num_or_acc(self.opts.ssize)

    ## instantiate converge handlers
    #converges = []
    #for limit in self.limits_from_option:
      #converges.append(progress.Converge(tol=tol, smooth=self.opts.smooth))

    #with progress.Abort() as abort, \
         #progress.Until(self.opts.until) as until, \
         #progress.Terminate() as term, \
         #progress.Monitor(formats=dict(max_acc='%.'+str(digits)+'f%%'),
                          #order=['min_count'],
                          #verbose=(self.opts.verbose and tol)) as mon, \
         #progress.Bar(ssize, text='calculate ALDOS',
                      #verbose=(self.opts.verbose and ssize)) as bar:

      ## as soon as the matrix has been rescaled for the first time and a
      ## rescaled energy array is retrieved from the core module, set this
      ## to True
      ## so basically this is True when at least one for-loop iteration has
      ## been done
      #scaleback = False

      ## get previous results to continue calculation
      #energs, denses, counts, accs, limits, eranges, vars, paramses \
      #= [], [], [], [], [], [], [], []
      #for outname in self.outnames:
        #if outname in self.infile:
          #energ  = self.infile[outname].x
          #dens   = self.infile[outname].y
          #count  = self.infile[outname].attrs.count
          #acc    = self.infile[outname].attrs.acc
          #limit  = self.infile[outname].attrs.limit
          #erange = self.infile[outname].attrs.erange
          #var    = self.infile[outname].attrs.var
          ##ci     = self.infile[outname].attrs.ci
          #params = self.infile[outname].attrs.params
        #else:
          #energ  = None
          #dens   = None
          #count  = 0
          #acc    = -1
          #limit  = None
          #erange = (None, None)
          #var    = None
          ##ci     = None
          #params = (None, None)
        #energs.append(energ)
        #denses.append(dens)
        #counts.append(count)
        #accs.append(acc)
        #limits.append(limit)
        #eranges.append(erange)
        #vars.append(var)
        #paramses.append(params)

      ## all energy ranges from the file have to be equal
      #for erange in eranges[1:]:
        #if eranges[0] != erange:
          #raise ValueError, 'all energy ranges from the files must be equal'

      ## get number of states per realizations of which the LDOS will be taken
      ## into account
      #spr = self.get_spr(self.opts.spr, self.infile.__scell__)

      ## initialize geometric average calculators
      #means = []
      #for outname, dens, count, var in \
      #zip(self.outnames, denses, counts, vars):
        #means.append(oalg.Mean(init_value=dens, init_count=count,
                               #init_var=var))

      ## the progress bar may jump ahead if a fixed sample size is given
      #if ssize and not abort.check() and not term.check() \
      #and not until.check():
        #min_count = min([mean.count for mean in means])
        #bar.jump(min_count)

      ## check truncation limit
      #for lind in xrange(len(limits)):
        #if limits[lind]:
          #if self.limits_from_option[lind] != limits[lind]:
            #raise ValueError, 'given truncation limit #%i' % (lind+1)+\
                              #'(%i) is ' % limits_from_option[lind]+\
                              #'incompatible with the one from the data '+\
                              #'file (%i)' % limits[lind]
        #else:
          #limits[lind] = self.limits_from_option[lind]

      ## cycle realizations
      #rcount = 0 # count realizations
      #while not until.check() and not abort.check() \
      #and not term.check():

        ## test if all converged or all have reached the requested sample size
        #is_converged, count_reached = [], []
        #for mean, converge in zip(means, converges):
          #is_converged.append(converge.check(mean.mean()) if tol else False)
          #count_reached.append(mean.count >= ssize if ssize else False)
        ##print 'is_converged', is_converged, len(is_converged)
        ##print 'count_reached', count_reached, len(count_reached)
        #if all(is_converged) or all(count_reached):
          ##print 'break1'
          #break

        ## get matrix of new disorder realization
        #mat, stateclasses = self.infile.__scell__.tbmat(distinguish=True)
        ##print 'stateclass lengths:', str([len(sc) for sc in stateclasses])
        #inds = self.get_stateclass(self.opts.stateclass, stateclasses)
        ##print self.opts.stateclass, inds
        #random.shuffle(inds) ### IS THIS CORRECT?
        #rcount += 1

        ## get energy range
        #### as all eranges are checked equal earlier, we would actually just
        #### need to compare with one of them
        #erange_from_option = self.get_erange(self.opts.erange, mat)
        #for erind, erange in enumerate(eranges):
          #if erange[0] and erange[1]:
            #if erange_from_option != erange:
              #raise ValueError, 'given energy range '+\
                                #'(%.2f, %.2f) ' % erange_from_option+\
                                #'is incompatible with that of the data '+\
                                #'file (%.2f, %.2f)' % erange
          #else:
            #eranges[erind] = erange_from_option

        ## rescale the matrix
        #mat, params = resc.rescale(mat, erange=eranges[0], omp=self.opts.omp)
        #### copy=False not working???

        ## cycle states
        #assert spr <= len(inds), 'spr value cannot be higher than #indices'
        #for sind in xrange(spr):
          ## check abort criterions
          #is_converged, count_reached = [], []
          #for mean, converge in zip(means, converges):
            #is_converged.append(converge.check(mean.mean()) if tol else False)
            #count_reached.append(mean.count >= ssize if ssize else False)
          #if until.check() or abort.check() or term.check() \
          #or all(is_converged) or all(count_reached):
            ##print 'break2'
            #break

          ## calculate LDOS
          ## erange isn't needed if rescaled is True
          #densities = ldosfunc(mat, limit=limits, rescaled=True,
                               #rcstr=self.opts.rcstr,
                               #intervals=self.opts.intervals,
                               #kernel=self.opts.kernel, state=inds[sind],
                               #num_threads=self.opts.num_threads)
          #energ = densities[0].x
          #scaleback = True

          ## add data to iterative averaging object
          ## scale density back to original energy spectrum (devide by "a")
          #for mean, density in zip(means, densities):
            #mean.add(density.y.real/params[0])

          ## report progress
          #min_count = min([mean.count for mean in means])
          #max_acc = max([converge.delta()*100 for converge in converges])
          #mon.update(min_count=min_count, max_acc=max_acc)
          #bar.step()

      ## scale energy back to original energy spectrum
      #if scaleback:
        #energ = resc.inverse_rescale(energ, params)
        #for eind in xrange(len(energs)):
          #energs[eind] = energ

      ## create new coFunc objects
      #for mean, energ, limit, erange, params, acc, outname in \
      #zip(means, energs, limits, eranges, paramses, accs, self.outnames):
        #if mean.count:
          #aldos = cofunc.coFunc(energ, mean.mean())
          #aldos.attrs.params    = params
          #aldos.attrs.tol       = tol
          #aldos.attrs.ssize     = ssize
          #aldos.attrs.spr       = spr
          #aldos.attrs.count     = mean.count
          #aldos.attrs.var       = mean.var()
          #aldos.attrs.ci        = mean.ci()
          #aldos.attrs.acc       = converge.delta() \
                                  #if converge.delta() != -1 else acc
          #aldos.attrs.rcount    = rcount
          #aldos.attrs.limit     = limit
          #aldos.attrs.erange    = erange
          #self.outfile[outname] = aldos

      ## finalize progress bar and progress monitor
      #mon.set_delay(0)
      #min_count = min([mean.count for mean in means])
      #max_acc = max([converge.delta()*100 for converge in converges])
      #mon.update(min_count=min_count, max_acc=max_acc)

    ## report that the timelimit has been reached
    #if until.check():
      #print 'Timelimit has been reached (%s).' % time.ctime(until.timestamp)

    ## report if the TERM signal has been received
    #if term.check():
      #print 'Process has been terminated'

    ## report if the process has been aborted by key press
    #if abort.check():
      #print 'Process has been aborted by key press'


#class _gldos_many(AveragingLDOS_KPMHDP00i):
  #"""Calculate geometric mean of local density of states (typical density
  #of states) for many different truncation numbers at once.

  #All results will be saved into the same file, named "gldos100", "gldos200",
  #etc. according to the chosen truncation numbers."""
  #__created__ = '2013-06-02'
  #__modified__ = '2013-06-19'
  ## based on tb.kpm2._aldos_many (developed 2013-06-01 until 2013-06-02)
  ## based on tb.kpm2._aldos (developed 2012-09-03 until 2012-10-09)
  ## based on tb.kpm._Ados (developed from 2012-04-26 until 2012-07-16) and
  ## on tb.kpm._Gdos (developed from 2011-12-20 until 2012-07-17)
  ## based on tb.kpm._Gdos from 2011-12-20 until 2012-04-18
  #innames = ['__scell__']
  #outnames = [] # names will be determined later (aldos100, aldos500, ...)
  #__autoinst__ = True

  #def prepare(self):
    ## get truncation limit
    ##size = self.infile.__scell__.size() # how to enable this?
    #self.limits_from_option = self.get_limits(self.opts.limit) # size

    ## determine outnames
    #self.outnames = ['gldos%i' % limit for limit in self.limits_from_option]

  #def main(self):
    ## make sure a supercell definition is given
    #if not isinstance(self.infile.__scell__, tb.sc.SuperCell):
      #raise ValueError, 'expecting supercell definition'

    ## instantiate LDOS HDP
    #ldosfunc = _ldos_many()

    ## understand sample size option
    #ssize, tol, digits = tb.misc.get_num_or_acc(self.opts.ssize)

    ## instantiate converge handlers
    #converges = []
    #for limit in self.limits_from_option:
      #converges.append(progress.Converge(tol=tol, smooth=self.opts.smooth))

    #with progress.Abort() as abort, \
         #progress.Until(self.opts.until) as until, \
         #progress.Terminate() as term, \
         #progress.Monitor(formats=dict(max_acc='%.'+str(digits)+'f%%'),
                          #order=['min_count'],
                          #verbose=(self.opts.verbose and tol)) as mon, \
         #progress.Bar(ssize, text='calculate GLDOS',
                      #verbose=(self.opts.verbose and ssize)) as bar:

      ## as soon as the matrix has been rescaled for the first time and a
      ## rescaled energy array is retrieved from the core module, set this
      ## to True
      ## so basically this is True when at least one for-loop iteration has
      ## been done
      #scaleback = False

      ## get previous results to continue calculation
      #energs, denses, counts, accs, limits, eranges, vars, paramses \
      #= [], [], [], [], [], [], [], []
      #for outname in self.outnames:
        #if outname in self.infile:
          #energ  = self.infile[outname].x
          #dens   = self.infile[outname].y
          #count  = self.infile[outname].attrs.count
          #acc    = self.infile[outname].attrs.acc
          #limit  = self.infile[outname].attrs.limit
          #erange = self.infile[outname].attrs.erange
          #var    = self.infile[outname].attrs.var
          ##ci     = self.infile[outname].attrs.ci
          #params = self.infile[outname].attrs.params
        #else:
          #energ  = None
          #dens   = None
          #count  = 0
          #acc    = -1
          #limit  = None
          #erange = (None, None)
          #var    = None
          ##ci     = None
          #params = (None, None)
        #energs.append(energ)
        #denses.append(dens)
        #counts.append(count)
        #accs.append(acc)
        #limits.append(limit)
        #eranges.append(erange)
        #vars.append(var)
        #paramses.append(params)

      ## all energy ranges from the file have to be equal
      ### what if the user want to add a new limit? It would be None first!
      #for erange in eranges[1:]:
        #if eranges[0] != erange:
          #raise ValueError, 'all energy ranges from the files must be equal'

      ## get number of states per realizations of which the LDOS will be taken
      ## into account
      #spr = self.get_spr(self.opts.spr, self.infile.__scell__)

      ## initialize geometric average calculators
      #means = []
      #for outname, dens, count, var in \
      #zip(self.outnames, denses, counts, vars):
        #means.append(oalg.gMean(init_value=dens, init_count=count,
                                #init_var=var))

      ## the progress bar may jump ahead if a fixed sample size is given
      #if ssize and not abort.check() and not term.check() \
      #and not until.check():
        #min_count = min([mean.count for mean in means])
        #bar.jump(min_count)

      ## check truncation limit
      #for lind in xrange(len(limits)):
        #if limits[lind]:
          #if self.limits_from_option[lind] != limits[lind]:
            #raise ValueError, 'given truncation limit #%i' % (lind+1)+\
                              #'(%i) is ' % limits_from_option[lind]+\
                              #'incompatible with the one from the data '+\
                              #'file (%i)' % limits[lind]
        #else:
          #limits[lind] = self.limits_from_option[lind]

      ## cycle realizations
      #rcount = 0 # count realizations
      #while not until.check() and not abort.check() \
      #and not term.check():

        ## test if all converged or all have reached the requested sample size
        #is_converged, count_reached = [], []
        #for mean, converge in zip(means, converges):
          #is_converged.append(converge.check(mean.mean()) \
                                             #if tol else False)
          #count_reached.append(mean.count >= ssize if ssize else False)
        ##print 'is_converged', is_converged, len(is_converged)
        ##print 'count_reached', count_reached, len(count_reached)
        #if all(is_converged) or all(count_reached):
          ##print 'break1'
          #break

        ## get matrix of new disorder realization
        #mat, stateclasses = self.infile.__scell__.tbmat(distinguish=True)
        ##print 'stateclass lengths:', str([len(sc) for sc in stateclasses])
        #inds = self.get_stateclass(self.opts.stateclass, stateclasses)
        ##print self.opts.stateclass, inds
        #random.shuffle(inds)
        #rcount += 1

        ## get energy range
        ### as all eranges are checked equal earlier, we would actually just
        ### need to compare with one of them
        #erange_from_option = self.get_erange(self.opts.erange, mat)
        #for erind, erange in enumerate(eranges):
          #if erange[0] and erange[1]:
            #if erange_from_option != erange:
              #raise ValueError, 'given energy range '+\
                                #'(%.2f, %.2f) ' % erange_from_option+\
                                #'is incompatible with that of the data '+\
                                #'file (%.2f, %.2f)' % erange
          #else:
            #eranges[erind] = erange_from_option

        ## rescale the matrix
        #mat, params = resc.rescale(mat, erange=eranges[0],
                                   #omp=self.opts.omp)
        ### copy=False not working?

        ## cycle states
        #assert spr <= len(inds), 'spr value cannot be higher than #indices'
        #for sind in xrange(spr):
          ## check abort criterions
          #is_converged, count_reached = [], []
          #for mean, converge in zip(means, converges):
            #is_converged.append(converge.check(mean.mean()) \
                                #if tol else False)
            #count_reached.append(mean.count >= ssize if ssize else False)
          #if until.check() or abort.check() or term.check() \
          #or all(is_converged) or all(count_reached):
            ##print 'break2'
            #break

          ## calculate LDOS
          ## erange isn't needed if rescaled is True
          #densities = ldosfunc(mat, limit=limits, rescaled=True,
                               #rcstr=self.opts.rcstr,
                               #intervals=self.opts.intervals,
                               #kernel=self.opts.kernel, state=inds[sind],
                               #num_threads=self.opts.num_threads)
          #energ = densities[0].x
          #scaleback = True

          #print densities[0].y

          ## add data to iterative averaging object
          ## scale density back to original energy spectrum (devide by "a")
          #for mean, density in zip(means, densities):
            #mean.add(density.y.real/params[0])

          ## report progress
          #min_count = min([mean.count for mean in means])
          #max_acc = max([converge.delta()*100 for converge in converges])
          #mon.update(min_count=min_count, max_acc=max_acc)
          #bar.step()

      ## scale energy back to original energy spectrum
      #if scaleback:
        #energ = resc.inverse_rescale(energ, params)
        #for eind in xrange(len(energs)):
          #energs[eind] = energ

      ## create new coFunc objects
      #for mean, energ, limit, erange, params, acc, outname in \
      #zip(means, energs, limits, eranges, paramses, accs, self.outnames):
        #if mean.count:
          #gldos = cofunc.coFunc(energ, mean.mean())
          #gldos.attrs.params    = params
          #gldos.attrs.tol       = tol
          #gldos.attrs.ssize     = ssize
          #gldos.attrs.spr       = spr
          #gldos.attrs.count     = mean.count
          #gldos.attrs.var       = mean.var()
          #gldos.attrs.ci        = mean.ci()
          #gldos.attrs.acc       = converge.delta() \
                                  #if converge.delta() != -1 else acc
          #gldos.attrs.rcount    = rcount
          #gldos.attrs.limit     = limit
          #gldos.attrs.erange    = erange
          #self.outfile[outname] = gldos

      ## finalize progress bar and progress monitor
      #mon.set_delay(0)
      #min_count = min([mean.count for mean in means])
      #max_acc = max([converge.delta()*100 for converge in converges])
      #mon.update(min_count=min_count, max_acc=max_acc)

    ## report that the timelimit has been reached
    #if until.check():
      #print 'Timelimit has been reached (%s).' % time.ctime(until.timestamp)

    ## report if the TERM signal has been received
    #if term.check():
      #print 'Process has been terminated'

    ## report if the process has been aborted by key press
    #if abort.check():
      #print 'Process has been aborted by key press'


#class _galdos(AveragingLDOS_KPMHDP00i):
  #"""Calculate both geometric (typical average) and arithmetic mean of the
  #local density of states at the same time, using each local density twice.
  #In this way, the numerical effort is easily reduced by a factor of 2 if
  #both types of averages are needed."""
  #__created__ = '2012-09-03'
  #__modified__ = '2013-06-20'
  ## based on tb.kpm2._gldos (developed 2012-09-03 until 2012-09-11)
  ## and tb.kpm2._aldos (developed 2012-09-03 until 2012-09-11)
  #innames = ['__scell__']
  #outnames = ['gldos', 'aldos']
  #__autoinst__ = True

  #def main(self):
    ## make sure that a supercell definition is given
    #if not isinstance(self.infile.__scell__, tb.sc.SuperCell):
      #raise ValueError, 'expecting supercell definition'

    ## instantiate LDOS HDP
    #ldosfunc = _ldos()

    ## understand sample size option
    #ssize, tol, digits = tb.misc.get_num_or_acc(self.opts.ssize)

    #with progress.Abort() as abort, \
         #progress.Converge(tol=tol, smooth=self.opts.smooth) as gconverge, \
         #progress.Converge(tol=tol, smooth=self.opts.smooth) as aconverge, \
         #progress.Until(self.opts.until) as until, \
         #progress.Terminate() as term, \
         #progress.Monitor(formats=dict(gacc='%.'+str(digits)+'f%%',
                                       #aacc='%.'+str(digits)+'f%%'),
                          #order=['gcount', 'gacc', 'acount', 'aacc'],
                          #verbose=(self.opts.verbose and tol)) as mon, \
         #progress.Bar(ssize, text='calculate GLDOS and ALDOS',
                      #verbose=(self.opts.verbose and ssize)) as bar:

      ## as soon as the matrix has been rescaled for the first time and a
      ## rescaled energy array is retrieved from the core module, set this
      ## to True
      ## so basically this is True when at least one for-loop iteration has
      ## been done
      #scaleback = False

      ## get previous results to continue calculation
      #if 'gldos' in self.infile:
        #energ  = self.infile.gldos.x
        #gdens   = self.infile.gldos.y
        #gcount  = self.infile.gldos.attrs.count
        #gacc    = self.infile.gldos.attrs.acc
        #glimit  = self.infile.gldos.attrs.limit
        #gerange = self.infile.gldos.attrs.erange
        #gvar    = self.infile.gldos.attrs.var
        #gparams = self.infile.gldos.attrs.params
      #else:
        #energ  = None
        #gdens   = None
        #gcount  = 0
        #gacc    = -1
        #glimit  = None
        #gerange = (None, None)
        #gvar    = None
        #gparams = (None, None)
      #if 'aldos' in self.infile:
        #energ  = self.infile.aldos.x
        #adens   = self.infile.aldos.y
        #acount  = self.infile.aldos.attrs.count
        #aacc    = self.infile.aldos.attrs.acc
        #alimit  = self.infile.aldos.attrs.limit
        #aerange = self.infile.aldos.attrs.erange
        #avar    = self.infile.aldos.attrs.var
        #aparams = self.infile.aldos.attrs.params
      #else:
        #energ  = None
        #adens   = None
        #acount  = 0
        #aacc    = -1
        #alimit  = None
        #aerange = (None, None)
        #avar    = None
        #aparams = (None, None)

      #if aparams != gparams:
        #raise ValueError, 'GLDOS and ALDOS in file %s ' \
                          #% self.infile.filename+\
                          #'have different rescaling parameters'
      #params = aparams

      ## get number of states per realizations of which the LDOS will be taken
      ## into account
      #spr = self.get_spr(self.opts.spr, self.infile.__scell__)

      ## initialize arithmetic and geometric average calculators
      #gmean = oalg.gMean(init_value=gdens, init_count=gcount,
                         #init_var=gvar)
      #amean = oalg.Mean(init_value=adens, init_count=acount,
                        #init_var=avar)

      ## the progress bar may jump ahead if a fixed sample size is given
      #if ssize and not abort.check() and not term.check() \
      #and not until.check():
        #bar.jump(min(gmean.count, amean.count))

      ## get truncation limit
      #size = self.infile.__scell__.size()
      #limit_from_option = self.get_limit(self.opts.limit, size)
      #if glimit:
        #if limit_from_option != glimit:
          #raise ValueError, 'given truncation limit '+\
                            #'(%i) is incompatible ' % limit_from_option+\
                            #'with the GLDOS limit from the data file '+\
                            #'(%i)' % glimit
      #if alimit:
        #if limit_from_option != alimit:
          #raise ValueError, 'given truncation limit '+\
                            #'(%i) is incompatible ' % limit_from_option+\
                            #'with the ALDOS limit from the data file '+\
                            #'(%i)' % alimit
      #limit = limit_from_option

      ## initialize intervals in case the loop will never be entered
      #params = (1., 0.)
      #intervals = self.get_intervals(self.opts.intervals, limit, params)

      ## cycle realizations
      #rcount = 0 # count realizations
      #while not until.check() and not abort.check() and not term.check() \
      #and (not tol or not gconverge.check(gmean.mean()) \
                   #or not aconverge.check(amean.mean())) \
      #and (not ssize or gmean.count < ssize or amean.count < ssize):
        ## get matrix of new disorder realization
        ##print time.ctime(), 'create matrix...'
        ##print 'sizeof(scell) before:', asizeof.asizeof(self.infile.__scell__)
        #mat, stateclasses = self.infile.__scell__.tbmat(distinguish=True,
                                                        #format='csr')
        ##print 'sizeof(scell) after:', asizeof.asizeof(self.infile.__scell__)
        ##print time.ctime(), 'matrix created'
        ##print time.ctime(), 'obtain stateclasses...'
        #inds = self.get_stateclass(self.opts.stateclass, stateclasses)
        ##print time.ctime(), 'stateclasses obtained'
        #random.shuffle(inds)
        #rcount += 1
        ##print mat.nnz, len(mat.data)

        ## get energy range
        #erange_from_option = self.get_erange(self.opts.erange, mat)
        #if gerange[0] and gerange[1]:
          #if erange_from_option != gerange:
            #raise ValueError, 'given energy range '+\
                              #'(%.2f, %.2f) ' % erange_from_option+\
                              #'is incompatible with the GLDOS erange from '+\
                              #'the data file (%.2f, %.2f)' % gerange
        #if aerange[0] and aerange[1]:
          #if erange_from_option != aerange:
            #raise ValueError, 'given energy range '+\
                              #'(%.2f, %.2f) ' % erange_from_option+\
                              #'is incompatible with the ALDOS erange from '+\
                              #'the data file (%.2f, %.2f)' % aerange
        #erange = erange_from_option

        ## rescale the matrix
        #mat, params = resc.rescale(mat, erange=erange, omp=self.opts.omp)
        #### copy=False not working???

        ## get number of intervals or stepwidth
        #intervals = self.get_intervals(self.opts.intervals, limit, params)

        ## cycle states
        #assert spr <= len(inds), 'spr value is waaay too high...'
        #for sind in xrange(spr):
          ## check abort criterions
          #if until.check() or abort.check() or term.check() \
          #or (tol and gconverge.check(gmean.mean())) \
          #or (tol and aconverge.check(amean.mean())) \
          #or (ssize and gmean.count >= ssize and amean.count >= ssize):
            #break

          ## calculate LDOS
          ## (erange isn't needed here, because rescaled is True)
          #energ, dens = ldosfunc(mat, limit=limit, rescaled=True,
                                 #rcstr=self.opts.rcstr, intervals=intervals,
                                 #kernel=self.opts.kernel, state=inds[sind],
                                 #num_threads=self.opts.num_threads).xy()
          #scaleback = True

          ## add data to iterative averaging objects
          ## scale density back to original energy spectrum (devide by "a")
          #try:
            #### not nice, but avoid negative values
            #gmean.add(abs(dens.real/params[0]))
          #except ValueError:
            #print >>sys.stderr, 'bug: dens.real =', dens.real
            #raise
          #amean.add(dens.real/params[0])

          ## report progress
          #mon.update(gcount=gmean.count, gacc=gconverge.delta()*100,
                     #acount=amean.count, aacc=aconverge.delta()*100)
          #bar.step()

      ## scale energy back to original energy spectrum
      #if scaleback:
        #energ = resc.inverse_rescale(energ, params)

      ## create new coFunc object
      #if gmean.count:
        #gldos = cofunc.coFunc(energ, gmean.mean())
        #gldos.attrs.params    = params
        #gldos.attrs.tol       = tol
        #gldos.attrs.ssize     = ssize
        #gldos.attrs.spr       = spr
        #gldos.attrs.count     = gmean.count
        #gldos.attrs.var       = gmean.var()
        #gldos.attrs.ci        = gmean.ci()
        #gldos.attrs.acc       = gconverge.delta() \
                                #if gconverge.delta() != -1 else gacc
        #gldos.attrs.rcount    = rcount
        #gldos.attrs.limit     = limit
        #gldos.attrs.erange    = gerange
        #gldos.attrs.intervals = intervals*params[0] \
                                #if isinstance(intervals, float) \
                                #else intervals
        #gldos.attrs.ndisc     = len(energ)
        #self.outfile.gldos = gldos
      #if amean.count:
        #aldos = cofunc.coFunc(energ, amean.mean())
        #aldos.attrs.params    = params
        #aldos.attrs.tol       = tol
        #aldos.attrs.ssize     = ssize
        #aldos.attrs.spr       = spr
        #aldos.attrs.count     = amean.count
        #aldos.attrs.var       = amean.var()
        #aldos.attrs.ci        = amean.ci()
        #aldos.attrs.acc       = aconverge.delta() \
                                #if aconverge.delta() != -1 else aacc
        #aldos.attrs.rcount    = rcount
        #aldos.attrs.limit     = limit
        #aldos.attrs.erange    = aerange
        #aldos.attrs.intervals = intervals*params[0] \
                                #if isinstance(intervals, float) \
                                #else intervals
        #aldos.attrs.ndisc     = len(energ)
        #self.outfile.aldos = aldos

      ## finalize progress bar and progress monitor
      #mon.set_delay(0)
      #mon.update(gcount=gmean.count, gacc=gconverge.delta()*100,
                  #acount=amean.count, aacc=aconverge.delta()*100)

    ## report that the timelimit has been reached
    #if until.check():
      #print 'Timelimit has been reached (%s).' % time.ctime(until.timestamp)

    ## report if the TERM signal has been received
    #if term.check():
      #print 'Process has been terminated'

    ## report if the process has been aborted by key press
    #if abort.check():
      #print 'Process has been aborted by key press'


#class KPMHDP(hdp.BaseHDP):
  #"""Define shared properties of HDPs that implement or utilize KPM
  #algorithms."""
  #__created__ ='2012-08-25'
  #__modified__ = '2013-06-20'
  ## former tb.kpm.KPMHDP (developed 2011-11-28 until 2012-08-15)
  #usage = '%prog [options] datafile [datafile2 [datafile3 ...]]'
  #epilog = """To do:
  #--> Use fast cosine transform for reconstruction (FCT, "divide-and-conquer"
      #algorithm). Just imagine O(N*log(N)) instead of O(N**2)..."""

  #def options(self):
    #self.add_option('-e', '--erange', dest='erange', default=None, type=str,
                  #help='set energy range. If None, use Lanczos algorithm')
    #self.add_option('-k', '--kernel', default='jackson',
                    #help='select kernel')
    ##self.add_option('-r', '--rcstr', default='dct',
                    ##help='select reconstruction method')
    #self.add_option('-l', '--limit', dest='limit', default='100', type=str,
                  #help='set truncation limit. If an integer is given, '+\
                       #'it defines a constant limit. If a float is '+\
                       #'given, or if the string ends with a percentage '+\
                       #'sign (%), it defines the desired accuracy for '+\
                       #'some convergence criterion. If an expression '+\
                       #'containing a slash (/) is given, it defines '+\
                       #'the desired ratio between truncation number '+\
                       #'and system size, which will be obtained from '+\
                       #'the supercell definition '+\
                       #'inside the file (in shell mode)')
    #self.add_option('-o', '--omp', default=False, action='store_true',
                    #help='use OpenMP parallelized algorithms (where '+\
                         #'possible)')
    #self.add_option('-n', '--num-threads', dest='num_threads', default=None,
                  #type=int,
                  #help='if not None, set number of OpenMP threads to use. '+\
                       #'If smaller than 1, choose automatically '+\
                       #'according to the number of processor cores')
    #self.add_option('-u', '--until', default='',
                    #help='confine execution time')
    #self.add_option('-i', '--intervals', default=None, type=str,
                    #help='set number of energy intervals. '+\
                         #'If None, use twice the truncation limit. '+\
                         #'If a float is given (a string including a point '+\
                         #'"."), the number is interpreted as the width of '+\
                         #'each energy interval ("Delta E"). The number of '+\
                         #'intervals is then determined based on this '+\
                         #'Delta E and the energy range specified by '+\
                         #'--erange')

  #def prepare(self):
    #if self.opts.omp:
      #misc.set_num_threads(self.opts.num_threads)

  #@staticmethod
  #def get_limit(limitopt, size=None):
    #"""Interprete given option string "limitopt". If a ratio is given, this
    #function needs the system size "size" (the ratio size/limit will be fixed
    #then)."""
    #__created__ = '2012-08-25'
    #__modified__ = '2012-09-04'
    ## former tb.kpm.KPMHDP.get_limit (developed 2011-11-29 until 2012-07-16)
    #if isinstance(limitopt, basestring) and '/' in limitopt:
      ## interprete as ratio between truncation number and system size

      ## make sure the division will result in a float (even if only integers
      ## are involved in the given expression)
      #if not '.' in limitopt:
        #limitopt += '.'

      ## evaluate expression
      #ratio = eval(limitopt)

      ## get system size
      #if size is None:
        #raise ValueError, 'system size must be known if a constant ratio '+\
                          #'system size/truncation limit is needed'
      #size = int(size)

      ## return limit
      #return int(round(ratio*size))
    #else:
      #return int(limitopt)

  #@staticmethod
  #def get_limits(limitopt, size=None):
    #"""Interprete given option string "limitopt". If a ratio is given, this
    #function needs information about the the system size (the ratio
    #size/limit will be kept constant then). This version of the function
    #can handle multiple comma-separated values, and will return a list of
    #truncation limits as a result."""
    #__created__ = '2013-05-31'
    #__modified__ = '2013-06-02'
    ## former tb.kpm.KPMHDP.get_limit (developed 2011-11-29 until 2012-07-16)
    ## based on tb.kpm.KPMHDP.get_limit (2012-08-25 until 2012-09-04)
    #if isinstance(limitopt, basestring):
      #values = []
      #for element in limitopt.split(','):
        #if '/' in element:
          ## interprete as ratio between truncation number and system size

          ## make sure the division will result in a float (even if only
          ## integers are involved in the given expression)
          #if not '.' in element:
            #element += '.'

          ## evaluate expression
          #ratio = eval(element)

          ## get system size
          #if size is None:
            #raise ValueError, 'system size must be known if a constant '+\
                              #'ratio system size/truncation limit is '+\
                              #'requested'
          #size = int(size)

          ## return limit
          #values.append(int(round(ratio*size)))
        #else:
          #values.append(int(element))
    #else:
      #values = [int(value) for value in limitopt]

    ## remove double entries, sort, and return
    #values = list(set(values))
    #values.sort()
    #return values

  #def get_erange(self, erange_option, mat):
    #"""Get and check energy range. Can not be done before the matrix is
    #loaded, so this is why this convenience method exists."""
    #__created__ = '2012-09-04'
    #__modified__ = '2012-09-04'
    ## copied from tb.kpm.KPMHDP.get_erange (2011-11-29 until 2012-08-15)

    ## get and check option
    #erange = tb.misc.opt2mathlist(erange_option, dtype=float)
    #if len(erange) > 2:
      #self.op.error('bad energy range')

    ## determine energy range
    #if len(erange) > 1:
      ## get energy range
      #emin, emax = erange
    #elif len(erange) > 0:
      ## assume symmetric energy range
      #if erange[0] < 0:
        #emin = erange[0]
        #emax = -emin
      #else:
        #emax = erange[0]
        #emin = -emax
    #else:
      ## calculate highest and lowest eigenvalues using Lanczos algorithm
      #bar = progress.Bar(1, 'use Lanczos algorithm to find energy range',
                          #verbose=self.opts.verbose)
      #### is Lanczos not working on the Blackpearl cluster?
      #some_eigvals = scipy.sparse.linalg.eigs(mat, k=12, which='LM',
                                              #return_eigenvectors=False)
      #some_eigvals = some_eigvals.real
      #emin = min(some_eigvals)
      #emin -= abs(.1*emin) # Just add an extra 10 % to make sure no
                            ## eigenvalue lies outside the interval [-1, 1]
      #emax = max(some_eigvals)
      #emax += abs(.1*emax) # Just add an extra 10 % to make sure no
                            ## eigenvalue lies outside the interval [-1, 1]
      #bar.step()

    ## return energy range
    #return emin, emax

  #@staticmethod
  #def get_intervals(num_or_step, truncation_limit, rescale_params):
    #"""Get number of intervals or stepwidth. If an integer or long is given
    #(or a string not including a "."), return number of intervals as an
    #integer. If a float is given (or a string including a "."), return the
    #rescaled stepwidth according to the given rescale parameters. If None is
    #given, return the integer 2*truncation_limit."""
    #__created__ = '2013-06-20'
    #__modified__ = '2013-06-20'
    #if num_or_step is None:
      #return 2*truncation_limit
    #if isinstance(num_or_step, basestring):
      #if '.' in num_or_step:
        #num_or_step = float(num_or_step)
      #else:
        #num_or_step = int(num_or_step)
    #if isinstance(num_or_step, float):
      ## rescale
      #num_or_step = num_or_step/rescale_params[0]
      #return num_or_step
    #else:
      #num_or_step = int(num_or_step)
      #return num_or_step


##======================#
## Averaging algorithms #
##======================#


#class AveragingLDOS_KPMHDP00i(KPMHDP00i):
  #__created__ = '2012-09-11'
  #__modified__ = '2012-10-12'
  #__autoinst__ = False

  #def options(self):
    #self.add_option('-s', '--sample-size', dest='ssize', default='1',
                    #type=str,
                    #help='set sample size. If string contains a decimal '+\
                         #'point (.) or ends with a percent sign (%), '
                         #'with "ppm" (parts per million) '+\
                         #'or with "ppb" (parts per billion), '+\
                         #'request a certain relative accuracy (dynamic '+\
                         #'number of samples)')
    #self.add_option('-p', '--spr', dest='spr', default='1%', type=str,
                    #help='set number of states per realization of which '+\
                        #'the local density of states is taken into '+\
                        #'account. If a float is given, or if the string '+\
                        #'ends with a percent sign (%), '+\
                        #'set the number relative to the '+\
                        #'total number of states of the system (the rank '+\
                        #'of the tight binding matrix)')
    #self.add_option('-o', '--smooth', dest='smooth', default=100,
                    #help='set smoothness level for convergence criterion '+\
                        #'(average over the given number of convergence '+\
                        #'tests). Note that the program will make at '+\
                        #'least this number of iterations before it '+\
                        #'will actually check for convergence. '+\
                        #'Only in effect if a dynamic number of samples '+\
                        #'is chosen using the --sample-size option')
    #self.add_option('-c', '--stateclass', default='', type=str,
                    #help='restrict selection of states to certain state '+\
                         #'class(es). By default, select all')
    ##self.add_option('-r', '--rcstr', default='dct', type=str,
                    ##help='set reconstruction method (std, dct)')
    ##self.add_option('-i', '--intervals', default=None, type=int,
                    ##help='set number of energy intervals. '+\
                         ##'If None, use twice the truncation limit')


  #def get_stateclass(self, stateclass_option, stateclasses):
    #"""Return state indices according to the selected stateclass(es)."""
    #__created__ = '2012-09-04'
    #__created__ = '2012-09-07'
    ## copied from tb.kpm2._ldos.get_stateclass (2012-09-03 until 2012-09-04)
    #if stateclass_option is None or not stateclass_option:
      ## by default, return all available indices
      #return list(set(itertools.chain(*stateclasses)))
    #else:
      #classinds = stateclass_option.split(',')
      #indices = []
      #for classind in classinds:
        #indices += stateclasses[int(classind)]
      #return list(set(indices))


##====================================#
## helper to convert wrong error data #
## can be deleted later...            #
##====================================#


#class _convert_stderr(hdp.HDP00i):
  #"""Convert old (wrong) standard error (stderr) attributes of GLDOS and
  #ALDOS to new (correct) variance (var), and confidence interval
  #(ci). Will only convert if the attribute "stderr" is found."""
  #__created__ = '2013-06-18'
  #__modified__ = '2013-06-19'
  #__autoinst__ = True
  #indep = True
  #sfrom = ffrom = sto = fto = hdp.FILE

  #def options(self):
    #self.add_option('-n', '--names', default='gldos,aldos',
                    #help='set dataset names')
    #self.add_option('-f', '--force', default=False, action='store_true',
                    #help='silently ignore non-existent datasets')

  #def prepare(self):
    #for name in self.opts.names.split(','):
      #if not name in ['gldos', 'aldos']:
        #print >>sys.stderr, 'names may only include "gldos" and "aldos"'
        #return

  #def main(self):
    #for name in self.opts.names.split(','):
      #if not name in self.infile:
        #if self.opts.force:
          #continue
        #print >>sys.stderr, 'convert-stderr: dataset "%s" ' % name +\
                            #'not found in file "%s"' % self.infile.filename
        #return
      #dset = self.infile[name]
      #if not 'stderr' in dset.attrs:
        #if self.opts.force:
          #continue
        #print >>sys.stderr, 'convert-stderr: dataset "%s" ' % name +\
                            #'in file "%s" ' % self.infile.filename +\
                            #'does not have attribute attrs.stderr'
        #return

      ## convert
      #stderr = dset.attrs.stderr if name == 'aldos' \
                #else numpy.log(dset.attrs.stderr)
      #std = stderr*numpy.sqrt(dset.attrs.count-1)
      #var = std**2

      ## save
      #dset.attrs.var = var
      #dset.attrs.ci = (stderr, stderr)
      #del dset.attrs.stderr
      #self.infile[name] = dset # overwrite with new version


#class _convert_std(hdp.HDP00i):
  #"""Convert old standard deviation (std) attributes of GLDOS and ALDOS to
  #new variance (var), following the new guideline that only the sample
  #variance should be saves alongside the GLDOS and ALDOS data. Also
  #overwrites the confidence interval (ci). Conversion only takes place if the
  #dataset "std" is found among the attributes of GLDOS and/or ALDOS."""
  #__created__ = '2013-06-18'
  #__modified__ = '2013-06-19'
  #__autoinst__ = True
  #indep = True
  #sfrom = ffrom = sto = fto = hdp.FILE

  #def options(self):
    #self.add_option('-n', '--names', default='gldos,aldos',
                    #help='set dataset names')
    #self.add_option('-f', '--force', default=False, action='store_true',
                    #help='silently ignore non-existent datasets')

  #def prepare(self):
    #for name in self.opts.names.split(','):
      #if not name in ['gldos', 'aldos']:
        #print >>sys.stderr, 'names may only include "gldos" and "aldos"'
        #return

  #def main(self):
    #for name in self.opts.names.split(','):
      #if not name in self.infile:
        #if self.opts.force:
          #continue
        #print >>sys.stderr, 'convert-std: dataset "%s" ' % name +\
                            #'not found in file "%s"' % self.infile.filename
        #return
      #dset = self.infile[name]
      #if not 'std' in dset.attrs:
        #if self.opts.force:
          #continue
        #print >>sys.stderr, 'convert-std: dataset "%s" ' % name +\
                            #'in file "%s" ' % self.infile.filename +\
                            #'does not have attribute attrs.std'
        #return

      ## convert
      #var = dset.attrs.std**2
      #stderr = dset.attrs.std/numpy.sqrt(dset.attrs.count-1)

      ## save
      #dset.attrs.var = var
      #dset.attrs.ci = (stderr, stderr)
      #del dset.attrs.std
      #self.infile[name] = dset # overwrite with new version


def _get_stateclass(stateclass_option, stateclasses):
    """Return state indices according to the selected stateclass(es)."""
    if stateclass_option is None or not stateclass_option:
        # by default, return all available indices
        return list(set(itertools.chain(*stateclasses)))
    else:
        classinds = stateclass_option.split(',')
        indices = []
        for classind in classinds:
            indices.append(int(classind))
        return list(set(indices))


#def get_states(self, state_option, inds):
  #"""Get indices of the states selected by the state option. Choose from the
  #given index list "inds"."""
  ##former tb.kpm._Ldos.get_state (2011-11-29)

  ## get available indices
  #inds = list(inds)
  #inds.sort()
  #inds = scipy.array(inds)

  ## process state option
  #if isinstance(state_option, basestring) and state_option.startswith('r'):
    ## choose a given number of random states
    #nstates = int(state_option[1:])
    #assert nstates >= 0
    #sinds = list(inds[numpy.random.permutation(len(inds))[:nstates]])
    #sinds.sort()
  #elif state_option == ':':
    ## choose all states (calculate total density of states)
    #sinds = inds
  #elif tb.misc.isiterable(state_option):
    #sinds = list(state_option)
  #elif isinstance(state_option, (int, long)):
    #sinds = [int(state_option)]
  #else:
    #sinds = tb.misc.opt2ranges(state_option, upper=len(inds))

  ## check number of selected states
  #if len(sinds) < 1:
    #self.op.error('number of states must be positive integer')

  ## store and return indices
  #return list(sinds)


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


#def _get_spr(spr_option, scell):
  #"""Interprete spr option to get number of states per realizations of which
  #the LDOS will be taken into account. Needs the SuperCell instance."""
  ## 2013-07-23 until 2013-07-23
  ## based on tb.kpm2.KPMHDP.get_spr (developed 2012-09-04 until 2012-09-07)
  ## copied from tb.kpm._Gdos.get_spr (developed 2012-02-24 until 2012-07-17)
  #spr_option = str(spr_option)
  #if spr_option[-1] == '%':
    #percent = float(spr_option[:-1])
    #if percent < 0 or percent > 100:
      #op.error('bad spr option: %s. ' % spr_option +\
                    #'Percentage must be from interval [0, 100]')
    #spr = int(percent/100*scell.size()) # scell.nents()
  #elif '.' in spr_option:
    #ratio = float(spr_option[:-1])
    #if ratio < 0 or ratio > 1:
      #op.error('bad spr option: %s. ' % spr_option +\
                    #'Ratio must be from interval [0, 1]')
    #spr = int(ratio*scell.size())
  #else:
    #spr = int(spr_option)

  #if spr < 1:
    #spr = 1
  #return spr
