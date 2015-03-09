from __future__ import print_function
import brewer2mpl
import logging
import os

from astropy.table import Table
from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from .. import fileio
from .. import match
from .. import trilegal
from .. import graphics
from .. import utils
from .starpop import StarPop

logger = logging.getLogger(__name__)

__all__ = ['SimGalaxy']

class ExctinctionTable(object):
    def __init__(self, extinction_table):
        self.data = ascii.read(extinction_table)
    
    def column_fmt(self, column):
        return column.translate(None, '()-').lower()
    
    def keyfmt(self, Rv, logg, column):
        str_column = self.column_fmt(column)
        return 'rv{}logg{}{}intp'.format(Rv, logg, str_column).replace('.', 'p')
    
    def select_Rv_logg(self, Rv, logg):
        return list(set(np.nonzero(self.data['Rv'] == Rv)[0]) &
                    set(np.nonzero(self.data['logg'] == logg)[0]))
    
    def _interpolate(self, column, Rv, logg):
        inds = self.select_Rv_logg(Rv, logg)
        key_name = self.keyfmt(Rv, logg, column)
        self.__setattr__(key_name, interp1d(np.log10(self.data['Teff'][inds]),
                                            self.data[column][inds],
                                            bounds_error=False))
    
    def get_value(self, teff, column, Rv, logg):
        new_arr = np.zeros(len(teff))

        indxs = [np.nonzero(logg <= 2.75)[0], np.nonzero(logg > 2.75)[0]]
        logg_vals = [2., 4.5]

        for i, logg_val in enumerate(logg_vals):
            key_name = self.keyfmt(Rv, logg_val, column)
            if not hasattr(self, key_name):
                self._interpolate(column, Rv, logg_val)
            f = self.__getattribute__(key_name)
            new_arr[indxs[i]] = f(teff[indxs[i]])
        return new_arr


class SimGalaxy(StarPop):
    '''
    A class for trilegal catalogs (simulated stellar population)
    '''
    def __init__(self, trilegal_catalog):
        StarPop.__init__(self)
        self.base, self.name = os.path.split(trilegal_catalog)
        #data = fileio.readfile(trilegal_catalog, only_keys=only_keys)
        if trilegal_catalog.endswith('hdf5'):
            data = Table.read(trilegal_catalog, path='data')
        else:
            data = ascii.read(trilegal_catalog)
        self.key_dict = dict(zip(list(data.dtype.names),
                                 range(len(list(data.dtype.names)))))
        #self.data = data.view(np.recarray)
        self.data = data

    def burst_duration(self):
        '''calculate ages of bursts'''
        lage = self.data['logAge']
        self.burst_length, = np.diff((10 ** np.min(lage), 10 ** np.max(lage)))

    def load_ic_mstar(self):
        '''
        separate C and M stars, sets their indicies as attributes: icstar and
        imstar, will include artificial star tests (if there are any).

        Trilegal simulation must have been done with -l and -a flags.

        This is done using self.rec meaning use should be e.g.:
        self.ast_mag2[self.rec][self.icstar]

        Hard coded:
        M star: C/O <= 1, LogL >= 3.3 Mdot <=-5, and TPAGB flag
        C star: C/O >= 1, Mdot <=-5, and TPAGB flag
        '''
        if not hasattr(self, 'rec'):
            self.rec = np.arange(len(self.data['CO']))

        try:
            co = self.data['CO'][self.rec]
        except KeyError as e:
            logger.error('No AGB stars. Trilegal must be run with -a')
            raise e

        logl = self.data['logL'][self.rec]
        itpagb = trilegal.get_stage_label('TPAGB')
        stage = self.data['stage'][self.rec]

        self.imstar, = np.nonzero((co <= 1) & (logl >= 3.3) & (stage == itpagb))
        self.icstar, = np.nonzero((co >= 1) & (stage == itpagb))

    def cmd_by_stage(self, filt1, filt2, yfilt='I', xlim=None, ylim=None,
                     oneplot=True, inds=None):
        '''
        Diagnostic plot(s) for a trilegal catalog.
        Produces a CMD or many CMDs with points colored by evolutionary stage.
        (Trilegal must have been run with -l flag or
         there must exist self.data['stage'] and something similar to
         trilegal.get_stage_label)
        
        Parameters
        ----------
        filt1, filt2 : str, str
            filters from column headings (Assumes CMD xaxis = filt1-filt2)
        
        yfilt : str
            if 'I', plot filt2 on yaxis, otherwise filt1.
        
        xlim, ylim : list or tuple
            limits sent to ax.set_[x,y]lim()
        
        oneplot : bool
            if True, will overlay all stages on one plot.
            if False, will make a fig with an axes for each stage as well as a
            summary panel on the top left (identical to oneplot=True)
        inds : array of indices to slice the data (untested)
        
        Returns
        -------
        axs : matplotlib.axes._subplots.AxesSubplot or array of them.
        '''
        def addtext(ax, xlabel, ylabel, xlim=None, ylim=None, label=None,
                    col=None):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            if label is not None:
                ax.set_title(label, **{'color': col})
            
            if xlim is not None:
                ax.set_xlim(xlim)
            
            if ylim is not None:
                ax.set_ylim(ylim)
            else:
                ax.set_ylim(ax.get_ylim()[::-1])
    
            ax.legend(loc='best', numpoints=1, frameon=False)
            return

        stage = np.array(self.data['stage'], dtype=int)
        color = self.data[filt1] - self.data[filt2]
        xlabel = '${}-{}$'.format(filt1, filt2)
        
        if yfilt == 'I':
            ymag = filt2
        else:
            ymag = filt1
            
        mag = self.data[ymag]
        ylabel = '${}$'.format(ymag)

        if inds is not None:
            stage = stage[inds]
            color = color[inds]
            mag = mag[inds]

        ustage = np.unique(stage)
        nstage = len(ustage)
        cols = brewer2mpl.get_map('Paired', 'Qualitative', nstage).mpl_colors
        icols = np.array(cols)[stage]
        labels = trilegal.get_stage_label(ustage)
        
        if oneplot:
            fig, axs = plt.subplots()
            ax = axs
        else:
            fig, (axs) = graphics.setup_multiplot(nstage + 1)
            ax = axs[0, 0]

        ax.scatter(color, mag, c=icols, alpha=0.2)
        # couldn't figure out how to make scatter care about labels...
        if oneplot:
            [ax.plot(-99, -99, '.', c=cols[i], label=labels[i])
             for i in range(nstage)]
        addtext(ax, xlabel, ylabel, xlim=xlim, ylim=ylim)

        if not oneplot:
            for i, (ax, st) in enumerate(zip(axs.ravel()[1:], ustage)):
                ind, = np.nonzero(stage == st)
                if len(ind) == 0:
                    continue
                ax.plot(color[ind], mag[ind], '.', color=cols[i], mew=0,
                        label='$N={}$'.format(len(ind)))
                addtext(ax, xlabel, ylabel, xlim=xlim, ylim=ylim,
                        label=labels[i], col=cols[i])
        return axs

    def hist_by_attr(self, attr, bins=10, stage=None, slice_inds=None):
        '''
        histogram of attribute in self.data sliced by slice_inds and/or stage.
        '''
        data = self.data[attr]
        if stage is not None:
            istage_s = 'i%s' % stage.lower()
            if not hasattr(self, istage_s):
                self.all_stages(stage.lower())
            istage = self.__dict__[istage_s]
        else:
            istage = np.arange(data.size)

        if slice_inds is None:
            slice_inds = np.arange(data.size)

        inds = list(set(istage) & set(slice_inds))
        hist, bins = np.histogram(data[inds], bins=bins)

        return hist, bins

    def all_stages(self, *stages):
        '''
        add the indices of evolutionary stage(s) as an attribute i[stage]
        '''
        if stages is ():
            stages = trilegal.get_stage_label()
        for stage in stages:
            i = self.stage_inds(stage)
            self.__setattr__('i%s' % stage.lower(), i)
        return

    def stage_inds(self, name):
        '''
        indices where self.data['stage'] is [name] evolutionary stage
        see trilegal.get_stage_label
        '''
        assert 'stage' in self.data.keys(), 'no stages marked in this file'
        inds, = np.nonzero(self.data['stage'] == trilegal.get_stage_label(name))
        return inds

    def apply_extinction(self, extinction_table, Rv=3.1, *filters):
        assert np.sum(self.data['Av']) == 0., 'Will not convert Av, must run trilegal without Av set'
        etab = ExctinctionTable(extinction_table)
        fmt = '{}_rv{}'
        names = []
        data = []
        for filt in filters:
            column = 'A({})'.format(filt)
            Alambda = etab.get_value(self.data['logTe'], column, Rv,
                                     self.data['logg'])
            names.append(fmt.format(filt, Rv))
            data.append(Alambda)

        self.add_data(names, data)
        return data
    
    def lognormalAv(self, disk_frac, mu, sigma, fg=0, df_young=0, df_old=8,
                    age_sep=3):
        '''
        IN DEVELOPMENT
        Alexia ran calcsfh on PHAT data with:
        -dAvy=0 -diskav=0.20,0.385,-0.95,0.65,0,8,3
        MATCH README states:
          -diskAv=N1,N2,N3,N4 sets differential extinction law, which is treated as
            a foreground component (flat distribution from zero to N1), plus
            a disk component (lognormal with mu=N3 and sigma=N4) affecting a
            fraction of stars equal to N2.  The ratio of the star scale
            height to gas scale height is specified per time bin in the
            parameter file.  For small values (0 to 1), the effect is
            simple differential extinction.  For larger values, one will see
            some fraction of the stars (1-N2) effectively unreddened (those
            in front of the disk) and the remainder affected by the lognormal.
            N1 should be non-negative, N2 should fall between zero and 1, and
            N4 should be positive.
         -diskAv=N1,N2,N3,N4,N5,N6,N7 is identical to the previous selection,
            except that the ratio of star to disk scale height is N5 for
            recent star formation, N6 for ancient stars, and transitions
            with a timescale of N7 Gyr.  N5 and N6 should be non-negative, and
            N7 should be positive.
        -dAvy=0.5 sets max additional differential extinction for young stars only.
            For stars under 40 Myr, up to this full value may be added; there
            is a linear ramp-down with age until 100 Myr, at which point no
            differential extinction is added.  It is possible that the
            best value to use here could be a function of metallicity.  Note
            that if both -dAv and -dAvy are used, the extra extinction applied
            to young stars is applied to the first of the two flat
            distributions.


        '''
        #  N1 Flat distribution from zero to N1 [0.2]
        #  N2 disk fraction of stars with lognormal [0.385]
        #  N3 mu lognormal [-0.95]
        #  N4 sigma lognormal [0.65]
        #  N5 like N2 but for recent SFR [0]
        #  N6 like N2 but for ancient SFR  [8]
        #  N7 transition between recent and ancient (Gyr) [3]
        #  dAvy was run at 0, not implemented yet.
        from scipy.stats import lognorm
        N1 + lognorm(mu=N3, sigma=N4)
        pass

