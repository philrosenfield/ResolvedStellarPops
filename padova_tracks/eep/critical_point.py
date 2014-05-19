import os
import numpy as np


def inds_between_ptcris(track, name1, name2, sandro=True):
    '''
    returns the indices from [name1, name2)
    this is iptcri, not mptcri
    they will be the same inds that can be used in Track.data
    '''
    if sandro is True:
        # this must be added in Tracks.load_critical_points!
        inds = track.sptcri
    else:
        inds = track.iptcri

    try:
        first = inds[track.ptcri.get_ptcri_name(name1, sandro=sandro)]
    except IndexError:
        first = 0

    try:
        second = inds[track.ptcri.get_ptcri_name(name2, sandro=sandro)]
    except IndexError:
        second = 0

    inds = np.arange(first, second)
    return inds


class critical_point(object):
    '''
    class to hold ptcri data from Sandro's ptcri file and input eep_obj
    which tells which critical points of Sandro's to ignore and which new
    ones to define. Definitions of new eeps are in the Track class.
    '''
    def __init__(self, filename, eep_obj=None):
        self.load_ptcri(filename, eep_obj=eep_obj)
        self.base, self.name = os.path.split(filename)
        self.get_args_from_name(filename)

    def get_args_from_name(self, filename):
        '''
        god i wish i knew regex
        '''
        zstr = filename.split('_Z')[-1]
        self.Z = float(zstr.split('_')[0])
        ystr = filename.replace('.dat', '').split('_Y')[-1].split('_')[0]
        if ystr.endswith('.'):
            ystr = ystr[:-1]
        self.Y = float(ystr)

    def get_ptcri_name(self, val, sandro=True, hb=False):
        if sandro is True:
            search_dict = self.sandros_dict
        elif hb is True:
            search_dict = self.key_dict_hb
        else:
            search_dict = self.key_dict

        if type(val) == int:
            return [name for name, pval in search_dict.items()
                    if pval == val][0]
        elif type(val) == str:
            return [pval for name, pval in search_dict.items()
                    if name == val][0]

    def load_ptcri(self, filename, eep_obj=None):
        '''
        reads the ptcri*dat file. If there is an eep_obj, it will flag the
        missing eeps in the ptcri file and only read the eeps that match both
        the eep_list and the ptcri file.
        should be part of eep...
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        # the lines have the path name, and the path has F7.
        begin, = [i for i in range(len(lines)) if lines[i].startswith('#')
                  and 'F7' in lines[i]]

        # the final column is a filename.
        col_keys = lines[begin + 1].replace('#', '').strip().split()[3:-1]

        # useful to save what Sandro defined
        self.sandro_eeps = col_keys
        self.sandros_dict = dict(zip(col_keys, range(len(col_keys))))

        # ptcri file has filename as col #19 so skip the last column
        usecols = range(0, 18)
        # invalid_raise will skip the last rows that Sandro uses to fake the
        # youngest MS ages (600Msun).
        data = np.genfromtxt(filename, usecols=usecols, skip_header=begin + 2,
                             invalid_raise=False)
        self.data = data
        self.masses = data[:, 1]

        # ptcri has all track data, but this instance only cares about one mass.
        data_dict = {}
        for i in range(len(data)):
            str_mass = 'M%.3f' % self.masses[i]
            data_dict[str_mass] = data[i][3:].astype('int')

        self.data_dict = data_dict
        self.please_define = []
        self.please_define_hb = []

        if eep_obj is not None:
            self.key_dict = dict(zip(eep_obj.eep_list,
                                     range(len(eep_obj.eep_list))))
            self.please_define = [c for c in eep_obj.eep_list
                                  if c not in col_keys]

            if eep_obj.eep_list_hb is not None:
                self.key_dict_hb = dict(zip(eep_obj.eep_list_hb,
                                        range(len(eep_obj.eep_list_hb))))
                # there is no mixture between Sandro's HB eeps since there
                # are no HB eeps in the ptcri files. Define them all here.
                self.please_define_hb = eep_obj.eep_list_hb

            self.eep = eep_obj
        else:
            self.please_define = []
            self.key_dict = self.sandros_dict

    def load_sandro_eeps(self, track):
        try:
            mptcri = self.data_dict['M%.3f' % track.mass]
        except KeyError:
            print 'No M%.3f in ptcri.data_dict.' % track.mass
            return -1
        track.sptcri = \
            np.concatenate([np.nonzero(track.data.MODE == m)[0]
                                    for m in mptcri])
