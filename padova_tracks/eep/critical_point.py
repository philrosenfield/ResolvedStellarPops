from __future__ import print_function
import os
import numpy as np


class Eep(object):
    '''
    a simple class to hold eep data. Gets added as an attribute to
    critical_point class.
    The lengths are then used in match.py
    '''
    def __init__(self):
        '''hard coded default eep_list and lengths'''
        eep_list = ['PMS_BEG', 'PMS_MIN', 'PMS_END', 'MS_BEG', 'MS_TMIN',
                    'MS_TO', 'SG_MAXL', 'RG_MINL', 'RG_BMP1', 'RG_BMP2',
                    'RG_TIP', 'HE_BEG', 'YCEN_0.550', 'YCEN_0.500',
                    'YCEN_0.400', 'YCEN_0.200', 'YCEN_0.100', 'YCEN_0.005',
                    'YCEN_0.000', 'TPAGB']
        eep_lengths = [60, 60, 80, 199, 100, 100, 70, 370, 30, 400, 40, 150,
                       100, 60, 100, 80, 80, 80, 100]

        ihb = eep_list.index('HE_BEG')
        eep_list_hb = np.copy(eep_list[ihb:])
        eep_lengths_hb = np.copy(eep_lengths[ihb:])

        self.eep_list = eep_list
        self.nticks = eep_lengths
        self.eep_list_hb = eep_list_hb
        self.nticks_hb = eep_lengths_hb

        # usefull to check match compatibility
        ims = eep_list.index('MS_TO')
        trans = ihb - 1
        self.nlow = np.sum(eep_lengths[:ims])
        self.nhb = np.sum(eep_lengths_hb)
        self.nms = np.sum(eep_lengths[:trans])
        self.ntot = self.nms + self.nhb + eep_lengths[trans]


class critical_point(object):
    '''
    class to hold ptcri data from Sandro's ptcri file and input eep_obj
    which tells which critical points of Sandro's to ignore and which new
    ones to define. Definitions of new eeps are in the Track class.
    '''
    def __init__(self, filename, sandro=True):
        self.base, self.name = os.path.split(filename)
        self.load_ptcri(filename, sandro=sandro)
        self.get_args_from_name(filename)

    def get_args_from_name(self, filename):
        '''
        strip Z and Y and add to self must have format of
        ..._Z0.01_Y0.25...
        '''
        zstr = filename.split('_Z')[-1]
        self.Z = float(zstr.split('_')[0])
        ystr = filename.replace('.dat', '').split('_Y')[-1].split('_')[0]
        if ystr.endswith('.'):
            ystr = ystr[:-1]
        self.Y = float(ystr)

    def inds_between_ptcris(self, track, name1, name2, sandro=True):
        '''
        returns the indices from [name1, name2)
        this is iptcri, not mptcri (which start at 1 not 0)
        they will be the same inds that can be used in Track.data
        '''
        if sandro:
            # this must be added in Tracks.load_critical_points!
            inds = track.sptcri
        else:
            inds = track.iptcri

        try:
            first = inds[self.get_ptcri_name(name1, sandro=sandro)]
        except IndexError:
            first = 0

        try:
            second = inds[self.get_ptcri_name(name2, sandro=sandro)]
        except IndexError:
            second = 0

        inds = np.arange(first, second)
        return inds

    def get_ptcri_name(self, val, sandro=True, hb=False):
        '''
        given the eep number or the eep name return the eep name or eep number.
        '''
        if sandro:
            pdict = self.sandros_dict
        elif hb:
            pdict = self.key_dict_hb
        else:
            pdict = self.key_dict

        if type(val) == int:
            return [name for name, pval in pdict.items() if pval == val][0]
        elif type(val) == str:
            return [pval for name, pval in pdict.items() if name == val][0]

    def load_ptcri(self, filename, sandro=True):
        '''
        Read the ptcri*dat file.
        Initialize Eep
        Flag the missing eeps in the ptcri file.
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        # the lines have the path name, and the path has F7.
        begin, = [i for i in range(len(lines)) if lines[i].startswith('#')
                  and 'F7' in lines[i]]

        # the final column is a filename.
        all_keys = lines[begin + 1].replace('#', '').strip().split()
        col_keys = all_keys[3:-1]
        # ptcri file has filename as col #19 so skip the last column
        usecols = range(0, len(all_keys) - 1)
        # invalid_raise will skip the last rows that Sandro uses to fake the
        # youngest MS ages (600Msun).
        data = np.genfromtxt(filename, usecols=usecols, skip_header=begin + 2,
                             invalid_raise=False)
        self.data = data
        self.masses = data[:, 1]

        data_dict = {}
        for i in range(len(data)):
            str_mass = 'M%.3f' % self.masses[i]
            data_dict[str_mass] = data[i][3:].astype(int)

        self.data_dict = data_dict

        eep_obj = Eep()
        eep_list = eep_obj.eep_list
        eep_list_hb = eep_obj.eep_list_hb
        self.key_dict = dict(zip(eep_list, range(len(eep_list))))
        self.key_dict_hb = dict(zip(eep_list_hb, range(len(eep_list_hb))))

        if sandro:
            # loading sandro's eeps means they will be used for match
            self.sandro_eeps = col_keys
            self.sandros_dict = dict(zip(col_keys, range(len(col_keys))))
            self.please_define = [c for c in eep_list if c not in col_keys]

            # there is no mixture between Sandro's HB eeps since there
            # are no HB eeps in the ptcri files. Define them all here.
            self.please_define_hb = eep_obj.eep_list_hb

        self.eep = eep_obj

    def load_eeps(self, track, sandro=True):
        '''load the eeps from the ptcri file'''
        try:
            ptcri = self.data_dict['M%.3f' % track.mass]
        except KeyError:
            track.flag = 'No M%.3f in ptcri.data_dict.' % track.mass
            return track
        if sandro:
            track.sptcri = \
                np.concatenate([np.nonzero(track.data.MODE == m)[0]
                                for m in ptcri])
        else:
            track.iptcri = ptcri
        track.ptcri_file = os.path.join(self.base, self.name)
        return track

    def save_ptcri(self, tracks, filename=None, hb=False):
        '''save parsec2match ptcris in same format as sandro's'''

        if filename is None:
            filename = os.path.join(self.base, 'p2m_%s' % self.name)
            if hb:
                filename = filename.replace('p2m', 'p2m_hb')

        if hb:
            key_dict = self.key_dict_hb
        else:
            key_dict = self.key_dict
        sorted_keys, inds = zip(*sorted(key_dict.items(),
                                        key=lambda (k, v): (v, k)))

        header = '# critical points in F7 files defined by sandro, basti, and phil \n'
        header += '# i mass kind_track %s fname \n' % (' '.join(sorted_keys))
        with open(filename, 'w') as f:
            f.write(header)
            linefmt = '%2i %.3f 0.0 %s %s \n'
            for i, track in enumerate(tracks):
                if track.flag is not None:
                    print('save_ptcri skipping %s: %s' % (track.name, track.flag))
                    continue
                ptcri_str = ' '.join(['%5d' % p for p in track.iptcri])
                f.write(linefmt % (i+1, track.mass, ptcri_str,
                                   os.path.join(track.base, track.name)))
        #print('wrote %s' % filename)
