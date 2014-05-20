'''
padova track files
'''
from __future__ import print_function
import os
import numpy as np



class Track(object):
    '''
    Padova stellar evolutoin track object.
    '''
    def __init__(self, filename, ptcri=None, loud=False):
        (self.base, self.name) = os.path.split(filename)
        if loud is True:
            print(filename)
        self.load_track(filename)
        self.filename_info()
        self.mass = self.data.MASS[0]
        if self.mass >= 12:
            # for high mass tracks, the mass starts much larger than it is
            # for (age<0.2). The mass only correct at the beginning of the MS.
            # Rather than forcing a ptcri load, we read the mass from the title.
            self.mass = float(self.name.split('_M')[1].split('.PMS')[0])
        self.ptcri = ptcri
        test = np.diff(self.data.AGE) >= 0
        if False in test:
            print('Track has age decreasing!!', self.mass)
            bads, = np.nonzero(np.diff(self.data.AGE) < 0)
            print('offensive MODEs:', self.data.MODE[bads])

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
        self.MUc = 1. / (np.sum((self.data[xi[i]] / ai[i]) * (1 + qi[i])
                                 for i in range(len(xi))))
        return self.Muc

    def filename_info(self):
        '''
        I wish I knew regex...
        '''
        (pref, __, smass) = self.name.split('.PMS')[0].split('_')
        #self.__setattr__[]
        #get that into attrs: 'Z0.0002Y0.4OUTA1.74M2.30'
        Z, Ymore = self.name.split('Z')[1].split('Y')
        for i, y in enumerate(Ymore):
            if y == '.':
                continue
            try:
                float(y)
            except:
                break
        self.Z = float(Z)
        self.Y = float(Ymore[:i])
        return

    def load_track(self, filename):
        '''
        reads PMS file into a record array. Stores header as string self.header

        this could be optimized, right now it reads the file twice and loops
        through it twice...
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        begin_track, = [i for i, l in enumerate(lines) if 'BEGIN TRACK' in l]
        self.header = lines[:begin_track]
        col_keys = lines[begin_track + 1].replace('#', '').strip().split()
        begin_track_skip = 2
        skip_footer = len([i for i, l in enumerate(lines)
                           if 'Comp Time' in l or 'AGE EXCEEDS' in l
                           or 'carbon burning' in l or 'REACHED' in l])
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
