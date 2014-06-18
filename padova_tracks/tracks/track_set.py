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

        if inputs.match is False:
            self.tracks_base = os.path.join(inputs.tracks_dir, self.prefix)
        else:
            self.tracks_base = inputs.outfile_dir
            inputs.track_search_term = \
                                inputs.track_search_term.replace('PMS', 'dat')

        if inputs.hb_only is False:
            self.find_tracks(track_search_term=inputs.track_search_term,
                             masses=inputs.masses, match=inputs.match)

        if inputs.hb is True:
            self.find_tracks(track_search_term=inputs.hbtrack_search_term,
                             masses=inputs.masses, match=inputs.match,
                             hb=True)
        else:
            self.hbtrack_names = []
            self.hbtracks = []
            self.hbmasses = []
            
    def find_tracks(self, track_search_term='*F7_*PMS', masses=None, hb=False,
                    match=False):
        '''
        loads tracks or hb tracks and their masses as attributes
        can load subset if masses (list, float, or string) is set.
        If masses is string, it must be have format like:
        '%f < 40' and it will use masses that are less 40.
        '''
        track_names = np.array(fileIO.get_files(self.tracks_base,
                               track_search_term))

        assert len(track_names) != 0, \
            'No tracks found: %s/%s' % (self.tracks_base, track_search_term)
        ext = '.' + fileIO.split_file_extention(track_names[0])[1]
        mass = np.array([os.path.split(t)[1].split('_M')[1].split(ext)[0]
                         for t in track_names], dtype=float)
        cut_mass, = np.nonzero(mass <= max_mass)
        track_names = track_names[cut_mass][np.argsort(mass[cut_mass])]
        mass = mass[cut_mass][np.argsort(mass[cut_mass])]

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

        if hb is True:
            track_str = 'hb%s' % track_str
            mass_str = 'hb%s' % mass_str

        track_attr = '%ss' % track_str
        self.__setattr__('%s_names' % track_str, track_names[inds])
        self.__setattr__(track_attr, [Track(track, match=match)
                                      for track in track_names[inds]])
        self.__setattr__('%s' % mass_str,
                         ['%.3f' % t.mass for t in self.__getattribute__(track_attr)])
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
