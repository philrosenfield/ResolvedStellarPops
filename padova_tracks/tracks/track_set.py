'''
a container of track
'''
from __future__ import print_function
import os
import matplotlib.pylab as plt
import numpy as np

from ResolvedStellarPops.fileio import fileIO

from .track import Track
from .track_diag import TrackDiag
from ..eep.critical_point import critical_point

max_mass = 120.
td = TrackDiag()

class TrackSet(object):
    '''
    A class to load multiple track instances at once.
    '''
    def __init__(self, inputs=None):
        if inputs is None:
            self.prefix = ''
            return
        self.prefix = inputs.prefix

        if not inputs.match:
            self.tracks_base = os.path.join(inputs.tracks_dir, self.prefix)
        else:
            self.tracks_base = inputs.outfile_dir
            inputs.track_search_term = \
                                inputs.track_search_term.replace('PMS', '.dat')

        if inputs.agb:
            self.find_tracks(track_search_term=inputs.track_search_term,
                             agb=True)
        else:
            self.agbtrack_names = []
            self.agbtracks = []
            self.agbmasses = []

        if inputs.hb:
            self.find_tracks(track_search_term=inputs.hbtrack_search_term,
                             masses=inputs.masses, match=inputs.match,
                             hb=True)
        else:
            self.find_tracks(track_search_term=inputs.track_search_term,
                             masses=inputs.masses, match=inputs.match)
            self.hbtrack_names = []
            self.hbtracks = []
            self.hbmasses = []
            
    def find_masses(self, track_search_term, hb=False, agb=False):
        track_names = fileIO.get_files(self.tracks_base, track_search_term)
        fname, ext = fileIO.split_file_extention(track_names[0]) 
        if hb:
            track_names = np.array([t for t in track_names if 'HB' in t])
            # ...PMS.HB
            fname, ext2 = fileIO.split_file_extention(fname)
            ext = '.%s%s' % (ext2, ext)        
        else:
            # ...PMS
            track_names = np.array([t for t in track_names if not 'HB' in t])
            ext = '.' + ext
            mstr = '_M'
        if agb:
            # Paola's tracks agb_0.66_Z0.00010000_ ... .dat
            ext = '_Z'
            mstr = 'agb_'

        # mass array
        mass = np.array([os.path.split(t)[1].split(mstr)[1].split(ext)[0]
                         for t in track_names], dtype=float)
        # inds of the masses to use and the correct order
        cut_mass, = np.nonzero(mass <= max_mass)
        morder = np.argsort(mass[cut_mass])
        
        # reorder by mass
        track_names = track_names[cut_mass][morder]
        mass = mass[cut_mass][morder]
        return track_names, mass

    def find_tracks(self, track_search_term='*F7_*PMS', masses=None, hb=False,
                    match=False, agb=False):
        '''
        loads tracks or hb tracks and their masses as attributes
        can load subset if masses (list, float, or string) is set.
        If masses is string, it must be have format like:
        '%f < 40' and it will use masses that are less 40.
        '''
        
        track_names, mass = self.find_masses(track_search_term, hb=hb, agb=agb)
        
        assert len(track_names) != 0, \
            'No tracks found: %s/%s' % (self.tracks_base, track_search_term)

        # only do a subset of masses
        if masses is not None:
            if type(masses) == float:
                inds = [masses]
            elif type(masses) == str:
                inds = [i for i in range(len(mass)) if eval(masses % mass[i])]
            if type(masses) == list:
                inds = np.array([])
                for m in masses:
                    try:
                        inds = np.append(inds, list(mass).index(m))
                    except ValueError:
                        # this mass is missing
                        pass
        else:
            inds = np.argsort(mass)
        
        track_str = 'track'
        mass_str = 'masses'

        if hb:
            track_str = 'hb%s' % track_str
            mass_str = 'hb%s' % mass_str
        if agb:
            track_str = 'agb%s' % track_str
            mass_str = 'agb%s' % mass_str

        tattr = '%ss' % track_str
        self.__setattr__('%s_names' % track_str, track_names[inds])
        self.__setattr__(tattr, \
            [Track(t, match=match, agb=agb, hb=hb) for t in track_names[inds]])
        self.__setattr__('%s' % mass_str, \
            ['%.3f' % t.mass for t in self.__getattribute__(tattr)
                                 if t.flag is None])
        return

    def all_inds_of_eep(self, eep_name):
        '''
        get all the ind for all tracks of some eep name, for example
        want ms_to of the track set? set eep_name = point_c if sandro==True.
        '''
        inds = []
        for track in self.tracks:
            check = self.ptcri.load_sandro_eeps(track)
            if check == -1:
                inds.append(-1)
                continue
            eep_ind = self.ptcri.get_ptcri_name(eep_name)
            if len(track.sptcri) <= eep_ind:
                inds.append(-1)
                continue
            data_ind = track.sptcri[eep_ind]
            inds.append(data_ind)
        return inds
