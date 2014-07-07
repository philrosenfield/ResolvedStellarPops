from __future__ import print_function
import os
import numpy as np

from .. import fileio
from .starpop import StarPop
from ..tools import hla_galaxy_info, bens_fmt_galaxy_info

from ..angst_tables import angst_data

__all__ = ['Galaxy']


class Galaxy(StarPop):
    '''
    angst and angrrr galaxy object. data is a ascii tagged file with stages.
    '''
    def __init__(self, fname, filetype=None, hla=True, angst=True, ext=None,
                 band=None, photsys=None, trgb=np.nan, z=-99, Av=None, dmod=None,
                 filter1=None, filter2=None):
        '''
        I hate this init.
        TODO:
        make a file_type reader that the init calls.
        add IR stuff to angst_tables not down here to be read at each call.
        '''
        self.base, self.name = os.path.split(fname)
        StarPop.__init__(self)
        # name spaces
        self.trgb = trgb
        self.z = z
        self.Av = Av
        self.dmod = dmod
        self.photsys = photsys
        self.load_data(fname, filetype=filetype, hla=hla, angst=angst,
                       band=band, photsys=photsys, filter1=filter1,
                       filter2=filter2, ext=ext)

        # angst table loads
        if angst is True:
            self.comp50mag1 = angst_data.get_50compmag(self.target,
                                                       self.filter1)
            self.comp50mag2 = angst_data.get_50compmag(self.target,
                                                       self.filter2)
            if hasattr(self, 'filter3'):
                self.comp50mag3 = angst_data.get_50compmag(self.target,
                                                           self.filter3)
                self.comp50mag4 = angst_data.get_50compmag(self.target,
                                                           self.filter4)

            self.trgb_av_dmod()
            # Abs mag
        if self.dmod is not None and self.photsys is not None:
            self.convert_mag(dmod=self.dmod, Av=self.Av, target=self.target)

    def load_data(self, fname, filetype=None, hla=True, angst=True, ext=None,
                  band=None, photsys=None, filter1=None, filter2=None):

        if hla is True:
            self.survey, self.propid, self.target, filts, psys = hla_galaxy_info(fname)
            # photometry
            self.filter1, self.filter2 = filts.upper().split('-')
            self.photsys = psys.replace('-', '_')
        else:
            self.survey = ' '
            self.photsys = photsys
            if None in [filter1, filter2]:
                self.propid, self.target, self.filter1, self.filter2 = bens_fmt_galaxy_info(fname)
            else:
                self.propid = ''
                self.target = fname
                self.filter1 = filter1
                self.filter2 = filter2

        if filetype is None:
            self.data = fileio.readfile(fname)
            if not None in [self.filter1, self.filter2]:
                self.mag1 = self.data[self.filter1]
                self.mag2 = self.data[self.filter2]
            else:
                self.mag1 = np.nan
                self.mag2 = np.nan
            self.data = self.data.view(np.recarray)

        elif 'fits' in filetype:
            hdu = fileio.pyfits.open(fname)
            if photsys is not None:
                ext = self.photsys.upper().split('_')[0]
            else:
                cam = hdu[0].header['CAMERA']
                if cam == 'ACS':
                    if photsys is None:
                        self.photsys = 'acs_wfc'
                elif cam == 'WFPC2':
                    self.photsys = 'wfpc2'
                else:
                    print('I do not know the photsys.')
            self.data = hdu[1].data
            self.ra = self.data['ra']
            self.dec = self.data['dec']

            if filetype == 'fitstable':
                self.header = hdu[0].header
                if ext is None:
                    ext = self.header['CAMERA']
                if '-' in ext:
                    if 'ACS' in ext:
                        ext = 'ACS'
                    else:
                        ext = ext.split('-')[-1]
                if band.upper() == 'IR':
                    ext = band.upper()
                self.mag1 = self.data['mag1_%s' % ext]
                self.mag2 = self.data['mag2_%s' % ext]
                self.filters = [self.filter1, self.filter2]

            if filetype == 'fitsimage':
                # made to read holtmann data...
                # this wont work on ir filters.
                filts = [f for f in self.data.columns.names
                         if f.endswith('w') and f.startswith('f')]
                order = np.argsort([float(f.replace('f', '').replace('w', ''))
                                    for f in filts])
                self.filter1 = filts[order[0]].upper()
                self.filter2 = filts[order[1]].upper()
                self.mag1 = self.data[self.filter1]
                self.mag2 = self.data[self.filter2]
            hdu.close()

        elif filetype == 'tagged_phot':
            self.data = fileio.read_tagged_phot(fname)
            self.mag1 = self.data['mag1']
            self.mag2 = self.data['mag2']
            self.stage = self.data['stage']

        elif filetype == 'match_phot':
            if None in [filter1, filter2]:
                names = ['mag1', 'mag2']
            else:
                names = [filter1, filter2]
            self.data = np.genfromtxt(fname, names=names)
            self.mag1 = self.data[names[0]]
            self.mag2 = self.data[names[1]]

        elif filetype == 'm31brick':
            if band is None:
                raise ValueError('Must supply band, uv, ir, acs')

            hdu = fileio.pyfits.open(fname)
            self.data = hdu[1].data
            self.header = hdu[0].header
            ext = band
            mag1 = self.data['%s_mag1' % ext]
            mag2 = self.data['%s_mag2' % ext]
            inds = list(set(np.nonzero(np.isfinite(mag1))[0]) &
                        set(np.nonzero(np.isfinite(mag2))[0]))
            self.mag1 = mag1  # [inds]
            self.mag2 = mag2  # [inds]
            self.rec = inds
            self.ra = self.data['ra']
            self.dec = self.data['dec']

            hdu.close()

        elif filetype == 'agbsnap':
            data = fileio.pyfits.getdata(fname)
            self.propid, self.target, _, self.filter1, self.filter2, \
                self.filter3, self.filter4 = self.name.split('.')[0].split('_')

            self.filters = [self.filter1, self.filter2, self.filter3,
                            self.filter4]

            if hasattr(data, 'MAG1_ACS'):
                self.mag1 = data.MAG1_ACS
                self.mag2 = data.MAG2_ACS
            else:
                raise AttributeError('not sure what to do here, boss')

            self.mag3 = data.MAG3_IR
            self.mag4 = data.MAG4_IR
            self.ra = data.RA
            self.dec = data.DEC
            self.data = data
        else:
            raise TypeError('bad filetype')
        self.color = self.mag1 - self.mag2

    def trgb_av_dmod(self):
        '''
        returns trgb, av, dmod from angst table
        '''
        filters = ','.join((self.filter1, self.filter2))
        (self.trgb, self.Av, self.dmod) = \
            angst_data.get_tab5_trgb_av_dmod(self.target, filters)
        if hasattr(self, 'filter3'):
            filters = ','.join((self.filter3, self.filter4))
            (self.ir_trgb, self.Av, self.dmod) = \
                angst_data.get_tab5_trgb_av_dmod(self.target, filters)

    def __str__(self):
        out = """ {s.survey:s} data:
                    Prop ID: {s.propid:s},
                    Target: {s.target:s},
                    dmod: {s.dmod:g},
                    Av: {s.Av:g},
                    Filters: {s.filter1:s} - {s.filter2:s}
                    Camera: {s.photsys:s}
                    Z: {s.z:.4f}""".format(s=self)
        return out

    def cut_mag_inds(self, mag2cut, mag1cut=None):
        '''
        a simple function to return indices of magX that are brighter than
        magXcut.
        '''
        mag2cut, = np.nonzero(self.mag2 <= mag2cut)
        if mag1cut is not None:
            mag1cut, = np.nonzero(self.mag1 <= mag1cut)
            cuts = list(set(mag1cut) & set(mag2cut))
        else:
            cuts = mag2cut
        return cuts
