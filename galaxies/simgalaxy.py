from __future__ import print_function
import palettable
import logging
import os

from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import ResolvedStellarPops as rsp

from .. import fileio
from .. import trilegal
from .. import graphics
from .. import utils
from .starpop import StarPop

logger = logging.getLogger(__name__)

__all__ = ['SimGalaxy']

class ExctinctionTable(object):
    """
    Class to use data from Leo's extinction Tables. Format is e.g,
    #Teff logg Av Rv A(F275W) A(F336W) A(F475W) A(F814W) A(F110W) A(F160W) (F275W-F336W)0 E(F275W-F336W)
     3500 4.5 1.0 3.1   1.433   1.623   1.167   0.58    0.327   0.202   4.271 -0.190  
     3500 4.5 1.0 5.0   1.186   1.341   1.106   0.644   0.379   0.234   4.271 -0.155
     3500 2.0 1.0 3.1   1.076   1.61    1.157   0.577   0.322   0.202   4.093 -0.534  
     3500 2.0 1.0 5.0   0.981   1.338   1.1     0.642   0.373   0.234   4.093 -0.357 
    """
    def __init__(self, extinction_table):
        self.data = Table.read(extinction_table, format='ascii')
    
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
            #print('reading')
            data = Table.read(trilegal_catalog, format='ascii.commented_header',
                              guess=False)
            #print('read')
            #data = Table.read(trilegal_catalog, format='ascii')
            #data = rsp.fileio.readfile(trilegal_catalog)
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
        M star: C/O <= 1, LogL >= 3.3, and TPAGB flag
        C star: C/O >= 1 and TPAGB flag
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

    def apply_extinction(self, extinction_table, filters, Rv=3.1, Av=1.,
                         add_to_array=False):
        """
        Add extinction correction to a trilegal catalog filters as columns
        [filter]_rv[Rv]_av[Av].
        
        Made as a quick look at the effects of not knowing Rv.
        Basically a call to function ExtinctionTable.get_value see that class
        for details.
        """
        assert np.sum(self.data['Av']) == 0., \
            'Will not convert Av, must run trilegal without Av set'
        if type(filters) is str:
            filters = [filters]

        etab = ExctinctionTable(extinction_table)
        fmt = '{}_rv{}_av{}'
        names = []
        data = []
        for filt in filters:
            column = 'A({})'.format(filt)
            Alambda = etab.get_value(self.data['logTe'], column, Rv,
                                     self.data['logg'])
            names.append(fmt.format(filt, Rv, Av))
            data.append(Alambda * Av)

        if add_to_array:
            self.add_data(names, data)
        else:
            return {k:v for k,v in zip(filters, data)}
    
    def apply_dAv(self, dAv, filters, photsys, Av=1., dAvy=0.5):
        """
        no need to apply this, right now it's only A = Av + 0.5 * dAv
          -dAv=0,0,0 sets differential extinction law, which is treated as two
            flat distibutions.  The first flat distribution goes from Av=0
            to the first number specified.  The second flat distirubtion
            goes from the first to the third numbers specified, and contains
            the second number's fraction of the total stars.  For example,
            if one third of the stars should have zero differential extinction
            and the rest to have differential extinction values stretching
            between Av=0 and Av=1, the command would be -dAv=0,0.67,1
        
        assert np.sum(self.data['Av']) == 0., \
            'Will not convert Av, must run trilegal without Av set'
        if type(dAv) is str:
            # e.g., dAv = '0.0,0.67,1'
            da0 = map(float, dAv.split(','))
        elif type(dAv) is list:
            # e.g., dAv = [0.0, 0.67, 1]
            da0 = dAv
        elif type(dAv) is float:
            # e.g., dAv = 0.2
            da0 = [dAv, 0.0, dAv]
        assert len(DA0) == 2, print(apply_dAv.__doc__)
        # if the second dist has lower Av than the first or if the fraction for
        # the second distribution is less than 0% or more than 100%
        if (da0[2] < da0[0]) or (da0[1] < 0.0) or (da0[1] > 1.0):
            da0 = [dAv, 0.0, dAv]

        # first flat dist: 0 --> dav[0]
        # second flat dist: dav[0] --> dav[2] contains dav[1] fraction of stars
        """
        dAv *= dAvy

        if type(filters) is str:
            filters = [filters]

        Alambdas = [rsp.astronomy_utils.Av2Alambda(Av + dAv, photsys, filt)
                    for filt in filters]
        
        return Alambdas

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

