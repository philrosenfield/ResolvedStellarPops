from __future__ import print_function
from ..tracks import TrackSet
from .. import parsec2match
from ...fileio import InputParameters
import numpy as np

indict1 = {'tracks_dir': '/Users/phil/research/stel_evo/CAF09_S12D_NS/tracks/',
           'set_name': 'CAF09_S12D_NS', 'prefixs': 'all', 'hb': False,
           'masses': '0.75 <= %f <= 12', 'do_interpolation': False}

indict2 = {'tracks_dir': '/Users/phil/research/stel_evo/CAF09_V1.2S_M36/tracks/',
           'set_name': 'CAF09_V1.2S_M36', 'prefixs': 'all', 'hb': False,
           'masses': '0.75 <= %f <= 12', 'do_interpolation': False}


class CompareTrackSets(object):
    '''class to compare two sets of stellar evolution models'''
    def __init__(self, indict1, indict2):
        self.load_track_sets(indict1, indict2)

    def load_track_sets(self, indict1, indict2):
        '''load each track set'''
        default_dict = parsec2match.initialize_inputs()
        self.inputs = []
        self.set_names = []
        self.track_sets = []            
        inp1 = InputParameters(default_dict=default_dict)
        inp1.add_params(indict1)
        prefixs1 = parsec2match.set_prefixs(inp1)
        inp2 = InputParameters(default_dict=default_dict)
        inp2.add_params(indict2)
        prefixs2 = parsec2match.set_prefixs(inp2)            

        for i in range(15):
            inp2.prefix = prefixs2[i]
            inp1.prefix = prefixs1[i]
            ts1 = TrackSet(inputs=inp1)
            ts1.load_characteristics()
            ts2 = TrackSet(inputs=inp2)
            ts2.load_characteristics()
            np.max([np.max(ts1.tracks[i].data.LOG_L - ts2.tracks[i].data.LOG_L) for i in range(len(ts1.tracks))])
                #track_set = np.append(track_set, ts)
            self.track_sets = np.append(self.track_sets, track_set)
            self.set_names = np.append(self.set_names, inp.set_name)
            
        return

    def compare_summary(self):
        np.unique(compts.track_sets[1].Zs), np.unique(compts.track_sets[0].Zs)
        
        [m for m in compts.track_sets[0].masses if not m in compts.track_sets[1].masses]
        [m for m in compts.track_sets[1].masses if not m in compts.track_sets[0].masses]
        [i for i,m in enumerate(compts.track_sets[0].masses) if m in compts.track_sets[1].masses]
        inds1 = [i for i,m in enumerate(compts.track_sets[0].masses) if m in compts.track_sets[1].masses]
        inds2 = [i for i,m in enumerate(compts.track_sets[1].masses) if m in compts.track_sets[0].masses]
        compts.track_sets[0].masses[inds1] == compts.track_sets[1].masses[inds2]
        
        
'''print 'S12'
print [[[' '.join(('%g' % t.Z, '%g' % t.mass, '%g' % float(h.strip().split()[1])))
        for h  in t.header if ' ALFOV ' in h] for t in ts.tracks] for ts in s12_ts]


[[[(t.Z, '%g' % t.mass, '%g' % t.cov, '%g' % float(h.strip().split()[1]))
   for h  in t.header if ' ALFOV ' in h] for t in ts.tracks]
 for ts in cov_ts]
'''
