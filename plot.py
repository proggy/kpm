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
"""KPM-related plot functions."""
__created__ = '2013-07-25'
__modified__ = '2014-03-15'
# former tb.plot (developed 2011-12-06 until 2013-07-24)

import matplotlib.pyplot as plt
import dummy
import numpy

try:
    from frog import Frog, items_of
except ImportError:
    Frog = dummy.Decorator


# common frog configuration for all frogs defined in this module
optdoc = dict(err='plot errorbars')
outmap = {0: None}
#shortopts = dict()
#longopts = dict()
#opttypes = dict()


@Frog(inmap=dict(ldos='$0/ldos', label='$0'), outmap=outmap,
      optdoc=optdoc, last=plt.show)
def pldos(ldos, err=False, fmt='-', label=None):
    """Plot LDOS (local density of states). Expect :class:`cofunc.coFunc`
    object."""
    yerr = 2*numpy.sqrt(ldos.attrs.var/ldos.attrs.count) if err else None
    label = label+'/ldos' if label else None
    ax = plt.errorbar(ldos.x, ldos.y, yerr=yerr, fmt=fmt, label=label)
    plt.xlabel('energy')
    plt.ylabel('LDOS')
    plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.pldos')
    return ax


@Frog(inmap=dict(dos='$0/dos', label='$0'), outmap=outmap,
      optdoc=optdoc, last=plt.show)
def pdos(dos, err=False, fmt='-', label=None):
    """Plot DOS (density of states). Expect :class:`cofunc.coFunc` object."""
    yerr = 2*numpy.sqrt(dos.attrs.var/dos.attrs.count) if err else None
    label = label+'/dos' if label else None
    ax = plt.errorbar(dos.x, dos.y, yerr=yerr, fmt=fmt, label=label)
    plt.xlabel('energy')
    plt.ylabel('DOS')
    plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.pdos')
    return ax


@Frog(inmap=dict(ados='$0/ados', label='$0'), outmap=outmap,
      optdoc=optdoc, last=plt.show)
def pados(ados, err=False, fmt='-', label=None):
    """Plot ADOS (density of states). Expect :class:`cofunc.coFunc` object."""
    yerr = 2*numpy.sqrt(ados.attrs.var/ados.attrs.count) if err else None
    label = label+'/ados' if label else None
    ax = plt.errorbar(ados.x, ados.y, yerr=yerr, fmt=fmt, label=label)
    plt.xlabel('energy')
    plt.ylabel('ADOS')
    plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.pados')
    return ax


@Frog(inmap=dict(aldos='$0/aldos', label='$0'), outmap=outmap,
      optdoc=optdoc, last=lambda: plt.show())
def paldos(aldos, err=False, fmt='-', label=None):
    """Plot ALDOS (arithmetic mean of the local density of states). Expect
    :class:`cofunc.coFunc` object. If *err* is *True*, calculate the standard
    error (using :class:`cofunc.coFunc` attributes *var* and *count*) and plot
    errorbars."""
    yerr = 2*numpy.sqrt(aldos.attrs.var/aldos.attrs.count) if err else None
    label = label if label else None
    ax = plt.errorbar(aldos.x, aldos.y, yerr=yerr, fmt=fmt, label=label)
    plt.xlabel('energy')
    plt.ylabel('ALDOS')
    plt.legend()
    plt.gcf().canvas.set_window_title(__name__+'.paldos')
    return ax


@Frog(inmap=dict(aldoslist='$@/aldos', disvals='$@/param'), outmap=outmap,
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
    # 2014-01-09
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


@Frog(inmap=dict(gldos='$0/gldos', label='$0'), outmap=outmap,
      optdoc=optdoc, last=last)
def pgldos(gldos, err=False, fmt='-', color=None, label=None):
    """Plot GLDOS (geometric mean of the local density of states, or typical
    density of states). Expect :class:`cofunc.coFunc` object. If *err* is
    *True*, calculate the standard error (using :class:`cofunc.coFunc`
    attributes *var* and *count*) and plot errorbars."""
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


@Frog(inmap=dict(ados='$0/ados', gldos='$0/gldos', label='$0'),
      outmap=outmap, optdoc=optdoc, last=last)
def pagldos(ados, gldos, err=False, color=None, label=None):
    """Plot ADOS (average density of states) and GLDOS (geometric mean of the
    local density of states). Expect two :class:`cofunc.coFunc` objects.  If
    *err* is *True*, calculate the standard error (using :class:`cofunc.coFunc`
    attributes *var* and *count*) and plot errorbars."""
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


@Frog(inmap=dict(aldos='$0/aldos', gldos='$0/gldos', label='$0'),
      outmap=outmap, optdoc=optdoc, last=last)
def pgaldos(aldos, gldos, err=False, color=None, label=None):
    """Plot ALDOS (arithmetic mean of the local density of states) and GLDOS
    (geometric mean of the local density of states). Expect two
    :class:`cofunc.coFunc` objects.  If *err* is *True*, calculate the standard
    error (using :class:`cofunc.coFunc` attributes *var* and *count*) and plot
    errorbars."""
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


@Frog(inmap=dict(gldos_list='$@/gldos', aldos_list='$@/aldos'), outmap=outmap,
      optdoc=optdoc, last=last)
def pchecksigma(gldos_list, aldos_list, bins=20, color=None, label=None):
    """Investigate the fluctuation of the standard error of the geometric mean
    among independent calculations in form of a histogram."""
    # 2014-02-28
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


#class _pgldos_fss(hdp2.plot.PlotHDP):
  #"""Plot GLDOS data in dependence of system size for specific energies.

  #To do:
  #--> select x-data (total system size N or longitudinal system length L)
  #--> offer curve fitting
  #--> semi-logarithmic plot
  #--> select y-data (instead of GLDOS)"""
  #__created__ = '2012-11-13'
  #__modified__ = '2013-02-18'
  #__autoinst__ = True
  #innames = ['gldos']

  #def options(self):
    #self.add_option('-b', '--bars', default=False, action='store_true',
                    #help='show errorbars (if confidence intervals exist '+\
                         #'among the attributes of the datasets)')
    #self.add_option('-e', '--energ', default='0',
                    #help='set energies')
    #self.add_option('-l', '--log', default=False, action='store_true',
                    #help='use logarithmic scaling on the y-axis')
    #self.add_option('-y', '--ydata', default='gldos', help='select y-data')
    #self.add_option('-f', '--fit', default=False, action='store_true',
                    #help='do curve fitting on the data')
    #self.add_option('-m', '--model', default='a/x**p',
                    #help='set model function to fit the GLDOS data')
    #self.add_option('-g', '--guess', default=None,
                    #help='set initial guess for the model parameters')
    #self.add_option('-i', '--import', dest='imp', default=None,
                    #help='import modules used in the model definition')
    #self.add_option('-x', '--xdata', dest='mode', default='L',
                    #help='set x-data')
    #self.add_option('-a', '--acc', default=None, type=str,
                    #help='set minimum accuracy or minimum sample count. '+\
                         #'Filter out input files that do not meet this '+\
                         #'requirement')

  #def prepare(self):
    ## choose fit model
    #self.model = fit2.Model(self.opts.model, guess=self.opts.guess,
                            #modules=self.opts.imp)

  #def main(self):
    #if len(self.infiles) == 0:
      #return

    ## get energies
    #energs = [float(e) for e in self.opts.energ.split(',')]
    #energs.sort()
    #self.colors(len(energs)) # each=1+self.opts.fit

    ## sort input files by system size
    #self.infiles.sort(key=lambda f: scipy.prod(f.__param__['shape']))

    ## collect system sizes
    #sizes = [scipy.mean(f.__param__['shape']) for f in self.infiles]

    ## choose x-data
    #Lmode = self.opts.mode == 'L' \
            #or 'length'.startswith(self.opts.mode.lower())
    #Nmode = self.opts.mode == 'N' \
            #or 'size'.startswith(self.opts.mode.lower())

    ## filter files by accuracy
    #if self.opts.acc:
      #num, acc, digits = tb.misc.get_num_or_acc(self.opts.acc)
    #inputfiles = []
    #for f in self.infiles:
      #if not 'gldos' in f:
        #continue
      #if self.opts.acc:
        #if num:
          #if f.gldos.attrs.get('count', 0) >= num:
            #inputfiles.append(f)
        #elif acc:
          #if f.gldos.attrs.get('acc', 1) <= acc:
            #inputfiles.append(f)
        #else:
          #inputfiles.append(f)
      #else:
        #inputfiles.append(f)
    #inputfiles.sort(key=lambda f: f.filename)

    ## select data
    #if Lmode:
      #x = [f.__param__['shape'][0] for f in inputfiles]
    #elif Nmode:
      #x = [scipy.prod(f.__param__['shape']) for f in inputfiles]
    #else:
      #self.op.error('unknown x-data: %s' % self.opts.mode)

    #if self.opts.fit:
      ## apply FSS on every energy interval
      #parsets = []
      #parstds = []
      #for eind, evalue in enumerate(energs):
        ## make fit
        #y = [f.gldos(evalue, bounds_error=False, fill_value=0.) \
              #for f in inputfiles]
        #popt, pcov = self.model.apply(x, y)
        #parsets.append(popt)
        #parstds.append(list(scipy.sqrt(pcov.diagonal())))

    ## cycle energies, plot data points
    #for evalue in energs:
      ## collect GLDOS values (and confidence intervals) at this energy
      #denses = []
      #cis = ([], [])
      #for infile in inputfiles:
        #if self.opts.ydata in infile:
          #denses.append(infile[self.opts.ydata](evalue, bounds_error=False))
          #if 'ci' in infile[self.opts.ydata].attrs:
            #lower, upper = infile[self.opts.ydata].a2cf('ci')
            #cis[0].append(lower(evalue))
            #cis[1].append(upper(evalue))
          #else:
            #cis[0].append(scipy.nan)
            #cis[1].append(scipy.nan)
        #else:
          #denses.append(scipy.nan)

      ## plot
      #label = 'E=%.2g' % evalue
      #pyplot.errorbar(sizes, denses, fmt='.' if self.opts.fit else '.--',
                     #label=label, yerr=cis if self.opts.bars else None)

    ## cycle energies again, plot fitting curves
    #if self.opts.fit:
      ## adjust view
      #axis = list(pyplot.axis()) # remember current view
      #if self.opts.axis[0] is not None and self.opts.axis[0] < axis[0]:
        #axis[0] = self.opts.axis[0]
      #if self.opts.axis[1] is not None and self.opts.axis[1] > axis[1]:
        #axis[1] = self.opts.axis[1]
      #if self.opts.axis[2] is not None and self.opts.axis[2] < axis[2]:
        #axis[2] = self.opts.axis[2]
      #if self.opts.axis[3] is not None and self.opts.axis[3] > axis[3]:
        #axis[3] = self.opts.axis[3]
      #self.opts.axis = axis

      #for eind, evalue in enumerate(energs):
        #parset = parsets[eind]
        #parstd = parstds[eind]
        #x = scipy.linspace(axis[0], axis[1], 200)
        #y = []
        #for xv in x:
          #parset_with_x = [xv,]+list(parset)
          #y.append(self.model.func(*parset_with_x))
        #if self.opts.fit:
          #pairs = []
          #for parname, parvalue, std in zip(self.model.parnames, parset,
                                            #parstd):
            #pairs.append('%s=%.2g$\pm$%.2g' % (parname, parvalue, std))
          #label = ', '.join(pairs) % parsets[eind]
          #pyplot.plot(x, y, label=label)

    ## show legend if number of energies is larger than one or fitting curve
    ## is shown
    #if not self.opts.legend and not self.opts.nolegend:
      #if len(energs) > 1 or self.opts.fit:
        #self.opts.legend = True
      #else:
        #self.opts.nolegend = True

    ## propose some figure properties
    #if Nmode:
      #xlabel = 'N'
    #elif Lmode:
      #xlabel = 'L'
    #else:
      #xlabel = 'system size'
    #self.propose(xlabel=xlabel, title='pgldos-fss',
                 #ylabel=self.opts.ydata.upper())


##class _Pgdos(hdp.PlotHDP):
  ##"""Plot the typical (geometrically averaged) density of states (GDOS)."""
  ##__created__ = '2012-03-18'
  ##__modified__ = '2012-07-19'
  ### former tb._Pgdos from 2011-06-09

  ##def __init__(self):
    ##hdp.PlotHDP.__init__(self)

    ### set options
    ##self.op.add_option('-m', '--mode',  dest='mode',  default='E',
                       ##help='set mode (which quantity should be shown on ' +
                            ##'the x-axis?)')
    ##self.op.add_option('-a', '--ados',  dest='ados',  default=False,
                       ##action='store_true',
                       ##help='plot total density of states (ADOS) of the ' +
                            ##'largest given system (in energy mode)')
    ##self.op.add_option('-c', '--curve', dest='curve', default='',
                       ##help='do curve fitting, specify fit model')
    ##self.op.add_option('-e', '--energ', dest='energ', default='',
                       ##help='set energy (in all but energy mode). ' +
                            ##'Multiple comma-separated values may be given')
    ##self.op.add_option('-p', '--param', dest='param', default=False,
                       ##action='store_true',
                       ##help='show resulting parameters of curve fitting ' +
                            ##'in legend')

  ##def __call__(self, *args, **kwargs):
    ##import matplotlib.pyplot as plt
    ##import scipy.interpolate as ip
    ##from tb.misc import opt2list
    ##from itertools import izip
    ##from fit import Model
    ##import numpy as np
    ##hdp.PlotHDP.__call__(self, *args, **kwargs)

    ### load data
    ##self.load('gdos_energ', 'gdos_dens')

    ### define mode conditions
    ##Emode = self.opts.mode == 'E' \
            ##or 'energy'.startswith(self.opts.mode.lower())
    ##Lmode = self.opts.mode == 'L' \
            ##or 'length'.startswith(self.opts.mode.lower())
    ##Nmode = self.opts.mode == 'N' \
            ##or 'size'.startswith(self.opts.mode.lower())
    ##Wmode = self.opts.mode == 'W' \
            ##or 'disorder'.startswith(self.opts.mode.lower())

    ### check options
    ##assert not self.opts.ados or Emode, 'ADOS only available in energy mode'
    ##assert not self.opts.energ or not Emode, \
           ##'energy can only be set if not in energy mode'
    ##assert not self.opts.curve or not Emode, \
           ##'curve fitting only available if not in energy mode'
    ##assert not self.opts.param or self.opts.curve, \
           ##'can only show parameters of curve fitting if curve fitting is '+\
           ##'actually used'
    ##self.opts.energ = opt2list(self.opts.energ, dtype=float)

    ### check that ratio of truncation limit and system size is about equal in
    ### all input files
    ##self.checkratio()

    ### define and select fitting model functions
    ##if self.opts.curve:
      ##if self.opts.curve.startswith('p'):
        ##power = float(self.opts.curve[1:])
        ##model = Model(lambda x, p: p[0]/x**power+p[1],
                      ##strrep='f(x) = %.2f/x^p + %.2f', est=(1., -1.),
                      ##paramstr='a=%.2f  b=%.3f') # shifted inverse
      ##elif self.opts.curve == '3':
        ##model = Model(lambda x, p: p[0]/x**p[1]+p[2],
                      ##strrep='f(x) = %.2f/x^%.2f + %.2f', est=(1., 1.5, -1.),
                      ##paramstr='a=%.2f  p=%.2f  b=%.3f')
      ##elif self.opts.curve == '2':
        ##model = Model(lambda x, p: p[0]/x**p[1], strrep='f(x) = %.2f/x^%.2f',
                      ##est=(1., 1.5), paramstr='a=%.2f  p=%.2f')

    ### choose mode
    ##if Emode:
      ### simply show GDOS versus energy of each file, sort filenames
      ###for d, p, f in izip(self.din, self.pin, self.fin):
      ##for index in np.argsort(self.fin):
        ##d = self.din[index]
        ##p = self.pin[index]
        ##f = self.fin[index]
        ##plt.plot(d.gdos_energ, d.gdos_dens,
                 ##label=self.nicelabel(p, default=f))

      ### plot ADOS
      ##if self.opts.ados:
        ##msind = np.argmax([np.prod(p.shape) for p in self.pin])
        ##ados_energ, ados_dens = hdp.dat(self.fin[msind],
                                        ##name='ados_energ,ados_dens')
        ##plt.plot(ados_energ, ados_dens, 'k--', label='ADOS')
                    #self.nicelabel(p, default=f)+' (ADOS)'

      ### propose x-axis label
      ##if 'hop' in self.pin[0]:  # in case of tight binding models with
                                  # hopping t
        ##self.propose(xlabel=r'$E/t$')
      ##elif 'coup' in self.pin[0]:  # in case of Heisenberg model with
                                     # coupling J
        ##self.propose(xlabel=r'$E/J_0$')
      ##else:
        ##self.propose(xlabel=r'$E$')

    ##elif Lmode:
      ### show system edge-length dependence

      ### categorize input data files
      ##psets, inds = self.catinfiles('shape')

      ### create interpolation function objects
      ##interps = []
      ##for d in self.din:
        ##interps.append(ip.interp1d(d.gdos_energ, d.gdos_dens,
                                     #bounds_error=False, fill_value=0.))

      ### set number of colors
      ##self.colors(len(self.opts.energ)*len(psets))

      ### plot GDOS data
      ##for pset, ind in zip(psets, inds):
        ##pins = self.pin[ind]
        ##interp = [interps[i] for i in ind]
        ##for e in self.opts.energ:
          ##d = [intp(e) for intp in interp]
          ##x = [p.shape[0] for p in pins]
          ##plt.plot(x, d, 'd', label='%s E/t=%.2f'
                                #% (self.nicelabel(pset), e))

      ### curve fitting
      ##if self.opts.curve:
        ### cycle categories (parameter combinations)
        ##for pset, ind in zip(psets, inds):
          ##pins = self.pin[ind]
          ##interp = [interps[i] for i in ind]
          ##for e in self.opts.energ:
            ### Make fit using least squares algorithm
            ##d = [intp(e) for intp in interp]
            ##x = [p.shape[0] for p in pins]
            ##model.leastsq(x, d)

            ### Plot fit curve
            ##if self.opts.param:
              ##label = model.paramstr()
            ##else:
              ##label = None
            ##model.plot(fmt='--', xmin=self.opts.axis[0],
                            #xmax=self.opts.axis[1], label=label)

        ### propose x-axis title
        ##self.propose(xlabel='L')

    ##elif Nmode:
      ### show system size dependence

      ### categorize input data files
      ##psets, inds = self.catinfiles('shape')

      ### create interpolation function objects
      ##interps = []
      ##for d in self.din:
        ##interps.append(ip.interp1d(d.gdos_energ, d.gdos_dens,
                            #bounds_error=False, fill_value=0.))

      ### set number of colors
      ##self.colors(len(self.opts.energ)*len(psets))

      ### plot GDOS data
      ##for pset, ind in zip(psets, inds):
        ##pins = self.pin[ind]
        ##interp = [interps[i] for i in ind]
        ##for e in self.opts.energ:
          ##d = [intp(e) for intp in interp]
          ##x = [np.prod(p.shape) for p in pins]
          ##plt.plot(x, d, 'd', label='%s E/t=%.2f'
                            #% (self.nicelabel(pset), e))

      ### curve fitting
      ##if self.opts.curve:
        ### cycle categories (parameter combinations)
        ##for pset, ind in zip(psets, inds):
          ##pins = self.pin[ind]
          ##interp = [interps[i] for i in ind]
          ##for e in self.opts.energ:
            ### make fit using least squares algorithm
            ##d = [intp(e) for intp in interp]
            ##x = [np.prod(p.shape) for p in pins]
            ##model.leastsq(x, d)

            ### plot fit curve
            ##if self.opts.param:
              ##label = model.paramstr()
            ##else:
              ##label = None
            ##model.plot(fmt='--', xmin=self.opts.axis[0],
                                #xmax=self.opts.axis[1], label=label)

        ### propose x-axis title
        ##self.propose(xlabel='N')

    ##elif Wmode:
      ### show disorder dependence

      ### categorize input data files
      ##psets, inds = self.catinfiles('scale')

      ### create interpolation function objects
      ##interps = []
      ##for d in self.din:
        ##interps.append(ip.interp1d(d.gdos_energ, d.gdos_dens,
                            #bounds_error=False, fill_value=0.))

      ### set number of colors
      ##self.colors(len(self.opts.energ)*len(psets))

      ### plot GDOS data
      ##for pset, ind in zip(psets, inds):
        ##pins = self.pin[ind]
        ##interp = [interps[i] for i in ind]
        ##for e in self.opts.energ:
          ##d = [intp(e) for intp in interp]
          ##x = [p.scale for p in pins]
          ##plt.plot(x, d, 'd', label='%s E/t=%.2f'
                            #% (self.nicelabel(pset), e))

      ### curve fitting
      ##if self.opts.curve:
        ### cycle categories (parameter combinations)
        ##for pset, ind in zip(psets, inds):
          ##pins = self.pin[ind]
          ##interp = [interps[i] for i in ind]
          ##for e in self.opts.energ:
            ### make fit using least squares algorithm
            ##d = [intp(e) for intp in interp]
            ##x = [p.scale for p in pins]
            ##model.leastsq(x, d)

            ### plot fit curve
            ##if self.opts.param:
              ##label = model.paramstr()
            ##else:
              ##label = None
            ##model.plot(fmt='--', xmin=self.opts.axis[0],
                            #xmax=self.opts.axis[1], label=label)

        ### propose x-axis title
        ##self.propose(xlabel='W/t')

    ##else:
      ##self.op.error('unknown mode: %s' % self.opts.mode)

    ### propose y-axis label
    ##if 'hop' in self.pin[0]:
      ##self.propose(ylabel=r'GDOS $\rho_{\mathrm{typ}}(E)\, t$')
    ##elif 'coup' in self.pin[0]:
      ##self.propose(ylabel=r'GDOS $\rho_{\mathrm{typ}}(E)\, J_0$')
    ##else:
      ##self.propose(ylabel=r'GDOS $\rho_{\mathrm{typ}}(E)$')

    ### finish the plot
    ##return self.finish()


##class _Pados(hdp.PlotHDP):
  ##"""Plot the total (arithmetically averaged) density of states (ADOS)."""
  ##__created__ = '2012-05-11'
  ##__modified__ = '2012-07-18'
  ### based on tb.plot.pgdos from 2012-03-18 until 2012-04-18

  ##def __init__(self):
    ##hdp.PlotHDP.__init__(self)

  ##def __call__(self, *args, **kwargs):
    ##hdp.PlotHDP.__call__(self, *args, **kwargs)
    ##import matplotlib.pyplot as plt, itertools, numpy

    ### load data
    ##self.load('ados_energ', 'ados_dens')

    ### check that ratio of truncation limit and system size is about equal in
    ### all input files
    ##self.checkratio()

    ### cycle input files, sort by filename
    ##for index in numpy.argsort(self.fin):
      ##din = self.din[index]
      ##pin = self.pin[index]
      ##fin = self.fin[index]
      ##plt.plot(din.ados_energ, din.ados_dens,
               ##label=self.nicelabel(pin, default=fin))

    ### propose figure properties
    ##if 'hop' in self.pin[0]:
      ##self.propose(xlabel=r'$E/t$',
                    #ylabel='ADOS $\\rho_{\mathrm{av}}(E)\, t$')
    ##elif 'coup' in self.pin[0]:
      ##self.propose(xlabel=r'$E/J_0$',
                   ##ylabel='ADOS $\\rho_{\mathrm{av}}(E)\, J_0$')
    ##else:
      ##self.propose(xlabel=r'$E$', ylabel='ADOS $\\rho_{\mathrm{av}}(E)$')

    ### finish plot
    ##return self.finish()


##class _Pgados(hdp.PlotHDP):
  ##"""Plot the typical (geometrically averaged) and the total (arithmetically
  ##averaged) density of states (GDOS and ADOS) for comparison."""
  ##__created__ = '2012-05-11'
  ##__modified__ = '2012-07-18'
  ### based on tb.plot.pados from 2012-05-11

  ##def __init__(self):
    ##hdp.PlotHDP.__init__(self)

  ##def __call__(self, *args, **kwargs):
    ##hdp.PlotHDP.__call__(self, *args, **kwargs)
    ##import matplotlib.pyplot as plt, itertools, numpy

    ### load data
    ##self.load('gdos_energ', 'gdos_dens', 'ados_energ', 'ados_dens')

    ### check that ratio of truncation limit and system size is about equal in
    ### all input files
    ##self.checkratio()

    ### choose number of colors and use each color twice
    ##self.colors(len(self.fin), each=2)

    ### cycle input files, sort by filename
    ##for index in numpy.argsort(self.fin):
      ##din = self.din[index]
      ##pin = self.pin[index]
      ##fin = self.fin[index]
      ##plt.plot(din.gdos_energ, din.gdos_dens, '-.',
               ##label='GDOS '+self.nicelabel(pin, default=fin))
      ##plt.plot(din.ados_energ, din.ados_dens,
               ##label='ADOS '+self.nicelabel(pin, default=fin))

    ### propose figure properties
    ##if 'hop' in self.pin[0]:
      ##self.propose(xlabel=r'$E/t$',
                   ##ylabel='GDOS $\\rho_{\mathrm{typ}}(E)\, t$ '+\
                          ##'and ADOS $\\rho_{\mathrm{av}}(E)\, t$')
    ##elif 'coup' in self.pin[0]:
      ##self.propose(xlabel=r'$E/J_0$',
                   ##ylabel='GDOS $\\rho_{\mathrm{typ}}(E)\, J_0$ '+\
                          ##'and ADOS $\\rho_{\mathrm{av}}(E)\, J_0$')
    ##else:
      ##self.propose(xlabel=r'$E$',
                   ##ylabel='GDOS $\\rho_{\mathrm{typ}}(E)$ '+\
                          ##'and ADOS $\\rho_{\mathrm{av}}(E)$')
    ##self.propose(legend='on')

    ### finish plot
    ##return self.finish()


def repeat_items(sequence, n=2):
    """Return new sequence where each item of the old *sequence* repeats the
    given number *n* of times."""
    new = []
    for item in sequence:
        new += [item]*n
    return new
