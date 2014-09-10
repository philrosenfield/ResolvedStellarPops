'''
a container of track
'''
from __future__ import print_function
import os
import matplotlib.pylab as plt
import numpy as np

from ResolvedStellarPops.fileio import fileIO
from ResolvedStellarPops.utils import sort_dict

from .track import Track
from .track_diag import TrackDiag
from ..eep.critical_point import critical_point, Eep

max_mass = 1000.
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
        mstr = '_M'
        if hb:
            track_names = np.array([t for t in track_names if 'HB' in t])
            # ...PMS.HB
            fname, ext2 = fileIO.split_file_extention(fname)
            ext = '.%s.%s' % (ext2, ext)
        else:
            # ...PMS
            track_names = np.array([t for t in track_names if not 'HB' in t])
            ext = '.' + ext
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

    def all_inds_of_eep(self, eep_name, sandro=True):
        '''
        get all the ind for all tracks of some eep name, for example
        want ms_to of the track set? set eep_name = point_c if sandro==True.
        '''
        inds = []
        for track in self.tracks:
            check = self.ptcri.load_eeps(track, sandro=sandro)
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

    def _load_ptcri(self, ptcri_loc, sandro=True, hb=False, search_extra=''):
        '''load ptcri file for each track in trackset'''
        
        if sandro:
            search_term = 'pt*'
        else:
            search_term = 'p2m*'

        new_keys = []
        for i, track in enumerate(self.tracks):
            pt_search =  '%s*%s*Z%g_*' % (search_term, search_extra , track.Z)
            ptcri_files = fileIO.get_files(ptcri_loc, pt_search)
            
            if hb:
                ptcri_files = [p for p in ptcri_files if 'hb' in p]
            else:
                ptcri_files = [p for p in ptcri_files if not 'hb' in p]
            
            for ptcri_file in ptcri_files:
                new_key = os.path.split(ptcri_file)[1].replace('0.', '').replace('.dat', '').lower()
                if os.path.split(track.base)[1] in os.path.split(ptcri_file)[1]:
                    if not hasattr(self, new_key):
                        ptcri = critical_point(ptcri_file, sandro=sandro)
                    else:
                        ptcri = self.__getattribute__(new_key)
                    self.tracks[i] = ptcri.load_eeps(track, sandro=sandro)
                    
                    new_keys.append(new_key)
                    self.__setattr__(new_key, ptcri)
            
        self.__setattr__('ptcris', list(np.unique(new_keys)))
        return

    def track_summary(self, full=True):
        assert hasattr(self, 'tracks'), 'Need tracks loaded'
        assert hasattr(self, 'ptcris'), 'Need ptcris loaded'

        ptcri_name = self.__getattribute__('ptcris')[0]
        ptcri = self.__getattribute__(ptcri_name)
        if full:
            eep_name, _ = sort_dict(ptcri.key_dict)
            fmt = ' & '.join(eep_name) + ' \\\\ \n'
            for t in self.tracks:
                fmt += ' & '.join('{:.3g}'.format(i)
                                  for i in t.data.AGE[t.iptcri[t.iptcri>0]])
                fmt += ' \\\\ \n'
        return fmt

    
        