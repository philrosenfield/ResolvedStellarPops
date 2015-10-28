from __future__ import print_function
import os
import numpy as np
from astropy.table import Table
from astropy.io import fits
from .. import fileio

from .starpop import StarPop
from .asts import parse_pipeline
from ..angst_tables import angst_data
from ..astronomy_utils import mag2Mag

__all__ = ['Galaxy']


class Galaxy(StarPop):
    '''angst and angrrr galaxy object'''
    def __init__(self, fname):
        self.base, self.name = os.path.split(fname)
        StarPop.__init__(self)
        # name spaces
        self.load_data(fname)

    def load_data(self, fname):
        if fname.endswith('fits'):
            self.data = fits.getdata(fname)
        else:
            self.data = Table.read(fname)
        self.target, self.filters = parse_pipeline(fname)

    def trgb_av_dmod(self, filt):
        '''returns trgb, av, dmod from angst table'''
        return angst_data.get_tab5_trgb_av_dmod(self.target, filt)

    def check_column(self, column, loud=False):
        vomit = ''
        if loud:
            vomit = ', '.join(self.dtype.names)
        assert column.upper() in self.data.dtype.names, \
            '{} not found. {}'.format(column, vomit)

    def absmag(self, column, filt, photsys=None, dmod=None, Av=None):
        self.check_column(column)
        if dmod is None:
            _, av, dmod = self.trgb_av_dmod(filt)
            if Av is None:
                Av = av

        if photsys is None:
            if 'ACS' in column.upper():
                photsys = 'acs_wfc'
            elif 'IR' in column.upper():
                photsys = 'wfc3ir'

        return mag2Mag(self.data[column], filt, photsys=photsys, dmod=dmod, Av=Av)
