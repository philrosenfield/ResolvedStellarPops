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


class eep(object):
    '''
    a simple class to hold eep data. Gets added as an attribute to ptcri class.
    '''
    def __init__(self, eep_list, eep_lengths=None, eep_list_hb=None,
                 eep_lengths_hb=None):
        self.eep_list = eep_list
        self.nticks = eep_lengths
        self.eep_list_hb = eep_list_hb
        self.nticks_hb = eep_lengths_hb
