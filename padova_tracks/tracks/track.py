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
    def __init__(self, filename, match=False, agb=False, hb=False):
        '''
        filename [str] the path to the PMS or PMS.HB file
        '''
        
        (self.base, self.name) = os.path.split(filename)
        # will house error string(s)
        self.flag = None
        self.info = {}
        if match:
            self.load_match_track(filename)
            self.track_mass(hb=hb)
            self.filename_info()
        elif agb:
            self.load_agb_track(filename)
        else:
            self.load_track(filename)
            self.filename_info()            
            if self.flag is None:
                self.track_mass(hb=hb)
        
        if self.flag is None:
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

    def track_mass(self, hb=False):
        ''' choose the mass based on the physical track starting points '''
        try:
            good_age, = np.nonzero(self.data.AGE > 0.2)
        except AttributeError:
            # match tracks have log age
            good_age = [[0]]
        if len(good_age) == 0:
            self.flag = 'unfinished track'
            self.mass = self.data.MASS[-1]
            return self.mass
        self.mass = self.data.MASS[good_age[0]]
        
        ind = -1
        if hb:
            #extension is .PMS.HB
            ind = -2

        ext = self.name.split('.')[ind]
            
        fmass = float(self.name.split('_M')[1].split('.' + ext)[0])
        if self.mass >= 12:
            self.mass = fmass
        elif np.abs(self.mass - fmass) > 0.005:
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
        # get Z, Y into attrs: 'Z0.0002Y0.4OUTA1.74M2.30'
        '''
        Z, Ymore = self.name.split('Z')[1].split('Y')
        Y = ''
        for y in Ymore:
            if y == '.' or y.isdigit():
                Y += y
            else:
                break
        self.Z = float(Z)
        self.Y = float(Y)
        self.ALFOV, = np.unique([float(l.replace('ALFOV', '').strip())
                                 for l in self.header if ' ALFOV ' in l])
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
        reads PMS file into a record array. Stores header and footer as
        list of strings in self.header
        
        if no 'BEGIN TRACK' in file will add message to self.flags
        
        if no footers are found will add message to self.info
        
        footer strings to search for are hard coded in a list.
        add to this list by perhaps
        import os
        for root, dirs, filenames in os.walk(tracks_dir):
            for filename in filenames:
                if filename.endswith('PMS'):
                    lines = open(os.path.join(root, filename)).readlines()
                    print lines[-2:]
        '''

        footers = ['Comp', 'EXCEED', 'burning', 'REACHED', 'STOP', 'END']

        with open(filename, 'r') as infile:
            lines = infile.readlines()

        # find the header
        begin_track = -1
        for i, l in enumerate(lines):
            if 'BEGIN TRACK' in l:
                begin_track = i
                break

        self.header = lines[:begin_track]

        if begin_track == -1:
            self.data = np.array([])
            self.col_keys = None
            self.flag = 'load_track error: no begin track'
            return

        # find the footer assuming it's no longer than 3 lines (for speed)
        skip_footer = 0
        for l in lines[-3:]:
            skip_footer += len([f for f in footers if f in l])

        if skip_footer > 0:
            self.header.append(lines[-skip_footer:])
        else:
            self.info['load_track warning'] = \
                'No footer unfinished track? %s' % filename
        
        # find ndarray titles (column keys)
        begin_track += 1
        col_keys = lines[begin_track].replace('#', '').strip().split()
        
        begin_track += 1
        
        # extra line for tracks that have been "colored"
        if 'information' in lines[begin_track + 1]:
            begin_track += 1
            col_keys = self.add_to_col_keys(col_keys, lines[begin_track])

        # read data into recarray
        iend = len(lines) - skip_footer
        nrows = iend - begin_track
        dtype = [(c, float) for c in col_keys]

        data = np.ndarray(shape=(nrows,), dtype=dtype)
        for row, i in enumerate(range(begin_track, iend)):
            data[row] = tuple(lines[i].split())
        
        self.data = data.view(np.recarray)
        self.col_keys = col_keys

        return

    def load_agb_track(self, filename, cut=True):
        '''
        Read Paola's tracks.
        Cutting out a lot of information that is not needed for MATCH
        col_keys = ['MODE', 'status', 'NTP', 'AGE', 'MASS', 'LOG_Mdot',
                    'LOG_L', 'LOG_TE', 'Mcore', 'Y', 'Z', 'PHI_TP', 'CO']
        usecols = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        '''
        def find_thermal_pulses(ntp):
            ''' find the thermal pulsations'''
            uniq_tps, uniq_inds = np.unique(ntp, return_index=True)
            tps = np.array([np.nonzero(ntp == u)[0] for u in uniq_tps])
            return tps
        
        def find_quiessent_points(tps, phi):
            '''
            The quiescent phase is the the max phase in each TP,
            i.e., closest to 1'''
            if tps.size == 1:
                qpts = np.argmax(phi)
            else:
                qpts = np.unique([tp[np.argmax(phi[tp])] for tp in tps])
            return qpts
        
        col_keys = ['MODE', 'status', 'NTP', 'AGE', 'MASS', 'LOG_Mdot',
                    'LOG_L', 'LOG_TE', 'Mcore', 'Y', 'Z', 'PHI_TP', 'CO']
        usecols = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]

        f = open(filename)
        line1 = f.readline()
        line2 = f.readline()
        
        if 'information' in line2:
            line1 = line1.replace('L_*', 'LOG_L')
            line1 = line1.replace('step', 'MODE')
            line1 = line1.replace('T_*', 'LOG_TE') 
            line1 = line1.replace('M_*', 'MASS')
            line1 = line1.replace('age/yr', 'AGE')
            line1 = line1.replace('dM/dt', 'LOG_Mdot')
            line1 = line1.replace('M_c', 'Mcore')
            col_keys = line1.strip().replace('#', '').replace('lg', '').split()    
            col_keys = self.add_to_col_keys(col_keys, line2)
            usecols = list(np.arange(len(col_keys)))
        
        data = np.genfromtxt(filename, names=col_keys, usecols=usecols)
        self.data = data.view(np.recarray)

        if self.data.NTP.size == 1:
            self.flag = 'no abg tracks'
            return
        self.Z = self.data.Z[0]
        self.Y = self.data.Y[0]
        self.mass = self.data.MASS[0]
        self.col_keys = col_keys
        
        # The first line in the agb track is 1. This isn't a quiescent stage.
        self.data.PHI_TP[0] = -99.

        self.tps = find_thermal_pulses(self.data.NTP)
        self.qpts = find_quiessent_points(self.tps, self.data.PHI_TP)
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
        return the max and min of a column in self.data, inds to slice.
        '''
        arr = self.data[col]
        if inds is not None:
            arr = arr[inds]
        ma = np.max(arr)
        mi = np.min(arr)
        return (ma, mi)
