'''
padova track files
'''
from __future__ import print_function
import os
import numpy as np

class Track(object):
    '''
    Padova stellar evolution track object.
    '''
    def __init__(self, filename, match=False):
        '''
        filename [str] the path to the PMS or PMS.HB file
        '''
        (self.base, self.name) = os.path.split(filename)
        if match is True:
            self.load_match_track(filename)
        else:
            self.load_track(filename)

        self.filename_info()
        # will house error string(s)
        self.flag = None
        self.track_mass()
        self.check_track()

    def check_track(self):
        '''check if age decreases'''
        try:
            age = self.data.AGE
        except AttributeError:
            age = self.data.logAge
        test = np.diff(age) >= 0
        if False in test:
            print('Track.__init__: track has age decreasing!!', self.mass)
            bads, = np.nonzero(np.diff(age) < 0)
            try:
                print('offensive MODEs:', self.data.MODE[bads])
            except AttributeError:
                print('offensive inds:', bads)
        return

    def track_mass(self):
        ''' choose the mass based on the physical track starting points '''
        try:
            good_age, = np.nonzero(self.data.AGE > 0.2)
        except AttributeError:
            good_age = [[0]]
        if len(good_age) == 0:
            self.flag = 'unfinished track'
            self.mass = self.data.MASS[-1]
            return self.mass
        self.mass = self.data.MASS[good_age[0]]
        ext = self.name.split('.')[-1]
        fmass = float(self.name.split('_M')[1].split('.' + ext)[0])
        if self.mass >= 12:
            self.mass = fmass
        elif self.mass != fmass:
            print('filename has M=%.4f track has M=%.4f' % (fmass, self.mass))
            self.flag = 'inconsistent mass'
        return self.mass

    def calc_Mbol(self, z_sun=4.77):
        '''
        Uses Z_sun = 4.77 adds self.Mbol and returns Mbol
        '''
        self.Mbol = z_sun - 2.5 * self.data.LOG_L
        return self.Mbol

    def calc_logg(self):
        '''
        cgs constant is -10.616 adds self.logg and returns logg
        '''
        self.logg = -10.616 + np.log10(self.mass) + 4.0 * self.data.LOG_TE - \
            self.data.LOG_L
        return self.logg

    def calc_core_mu(self):
        '''
        Uses X, Y, C, and O.
        '''
        xi = np.array(['XCEN', 'YCEN', 'XC_cen', 'XO_cen'])
        ai = np.array([1., 4., 12., 16.])
        # fully ionized
        qi = ai/2.
        self.muc = 1. / (np.sum((self.data[xi[i]] / ai[i]) * (1 + qi[i])
                                 for i in range(len(xi))))
        return self.muc

    def filename_info(self):
        '''
        I wish I knew regex...
        '''
        #(pref, __, smass) = self.name.split('.PMS')[0].split('_')
        #self.__setattr__[]
        #get that into attrs: 'Z0.0002Y0.4OUTA1.74M2.30'
        Z, Ymore = self.name.split('Z')[1].split('Y')
        Y = ''
        for y in Ymore:
            if y == '.' or y.isdigit() is True:
                Y += y
            else:
                break
        self.Z = float(Z)
        self.Y = float(Y)
        return

    def load_match_track(self, filename):
        '''
        load the match interpolated tracks into a record array.
        the file contains Mbol, but it is converted to LOG_L on read.
        LOG_L = (4.77 - Mbol) / 2.5
        names = 'logAge', 'MASS', 'LOG_TE', 'LOG_L', 'logg', 'CO'
        '''
        self.col_keys = 'logAge', 'MASS', 'LOG_TE', 'LOG_L', 'logg', 'CO'
        data = np.genfromtxt(filename, names=self.col_keys,
                             converters={3: lambda m: (4.77 - float(m)) / 2.5})
        self.data = data.view(np.recarray)
        return data

    def load_track(self, filename):
        '''
        reads PMS file into a record array. Stores header as string self.header

        this could be optimized, right now it reads the file twice
        '''
        # possible strings you can find in the footer, add to this list as
        # Sandro adds more 
        footers = ['Comp Time', 'EXCEED', 'carbon burning', 'REACHED', 'STOP']
        
        with open(filename, 'r') as f:
            lines = f.readlines()

        # find the header and footer
        skip_footer = 0
        for i, l in enumerate(lines):
            if 'BEGIN TRACK' in l:
                begin_track = i
            skip_footer += len([f for f in footers if f in l])
        
        self.header = lines[:begin_track]
        if skip_footer > 0:
            self.header.append(lines[-skip_footer:])
        else:
            print('load_track warning: No footer unfinished track? %s'
                  % filename)

        col_keys = lines[begin_track + 1].replace('#', '').strip().split()
        begin_track_skip = 2
        
        # Hack to read tracks that have been "colored"
        if 'information' in lines[begin_track + 2]:
            col_keys = self.add_to_col_keys(col_keys, lines[begin_track + 2])
            begin_track_skip += 1

        data = np.genfromtxt(filename, skiprows=begin_track + begin_track_skip,
                             names=col_keys, skip_footer=skip_footer,
                             invalid_raise=False)

        self.data = data.view(np.recarray)
        self.col_keys = col_keys

        return

    def add_to_col_keys(self, col_keys, additional_col_line):
        '''
        If fromHR2mags was run, A new line "Additional information Added:..."
        is added, this adds these column keys to the list.
        '''
        new_cols = additional_col_line.split(':')[1].strip().split()
        col_keys = list(np.concatenate((col_keys, new_cols)))
        return col_keys

    def maxmin(self, col, inds=None):
        '''
        returns the max and min of a column in self.data. use inds to index.
        '''
        arr = self.data[col]
        if inds is not None:
            arr = arr[inds]
        ma = np.max(arr)
        mi = np.min(arr)
        return (ma, mi)
